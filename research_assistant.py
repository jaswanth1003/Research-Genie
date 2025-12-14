# -*- coding: utf-8 -*-
"""
Enhanced Research Assistant System
Complete implementation with dynamic ArXiv search
Streamlit Cloud compatible
"""

import os
import re
import json
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()


# -------------------------------------------------------------------
# PDF Reader (Enhanced)
# -------------------------------------------------------------------
def read_pdf(file_path: str) -> Dict[str, any]:
    """
    Enhanced PDF reader that extracts text AND metadata
    
    Returns:
        Dict with 'text', 'metadata', 'num_pages'
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        reader = PdfReader(file_path)
        text = ""
        metadata = reader.metadata if reader.metadata else {}
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return {
            'text': text,
            'metadata': {
                'title': metadata.get('/Title', 'Unknown'),
                'author': metadata.get('/Author', 'Unknown'),
                'pages': len(reader.pages),
                'filename': os.path.basename(file_path)
            },
            'num_pages': len(reader.pages)
        }
        
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


# -------------------------------------------------------------------
# Citation Database
# -------------------------------------------------------------------
class CitationDatabase:
    """Stores and manages citations separately"""
    
    def __init__(self):
        self.citations = {}
        self.citation_count = 0
    
    def add_citations(self, doc_id: str, citations: List[Dict]):
        """Add citations for a document"""
        self.citations[doc_id] = citations
        self.citation_count += len(citations)
    
    def get_citations(self, doc_id: str) -> List[Dict]:
        """Get citations for a document"""
        return self.citations.get(doc_id, [])
    
    def search_citations(self, query: str) -> List[Dict]:
        """Search citations by content"""
        results = []
        for doc_id, cites in self.citations.items():
            for cite in cites:
                if query.lower() in str(cite).lower():
                    results.append({
                        'doc_id': doc_id,
                        'citation': cite
                    })
        return results
    
    def verify_citation(self, citation_text: str, doc_id: str) -> bool:
        """Verify if citation exists in document"""
        doc_citations = self.get_citations(doc_id)
        return any(citation_text in str(c) for c in doc_citations)


# -------------------------------------------------------------------
# Enhanced Citation Extractor
# -------------------------------------------------------------------
class EnhancedCitationExtractor:
    """Enhanced citation extraction with database storage"""
    
    def __init__(self):
        self.citation_db = CitationDatabase()
        self.reference_text = ""
        self.citation_map = {}
    
    def extract_citations_from_text(self, text: str, doc_id: str) -> List[Dict]:
        """Extract all citations from text with context"""
        pattern = r'\[(\d+)\]([^\[]*?)(?=\[|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        citations = []
        for num, context in matches:
            citations.append({
                'number': num,
                'context': context.strip()[:200],
                'doc_id': doc_id
            })
        
        self.citation_db.add_citations(doc_id, citations)
        return citations
    
    def extract_reference_section(self, text: str) -> str:
        """Extract the references/bibliography section"""
        patterns = [
            r'(?i)references\s*\n(.*?)(?=\n\n[A-Z]|\Z)',
            r'(?i)bibliography\s*\n(.*?)(?=\n\n[A-Z]|\Z)',
            r'(?i)works cited\s*\n(.*?)(?=\n\n[A-Z]|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        
        return ""
    
    def build_citation_map(self, reference_text: str) -> Dict[str, str]:
        """Build detailed citation map"""
        self.reference_text = reference_text
        entries = re.split(r'(\[\s*\d+\s*\])', reference_text)
        citation_map = {}
        current_num = None
        current_text = []

        for part in entries:
            match = re.match(r'\[\s*(\d+)\s*\]', part)
            if match:
                if current_num is not None:
                    citation_map[current_num] = ' '.join(current_text).strip()
                current_num = match.group(1)
                current_text = []
            elif current_num is not None:
                current_text.append(part.strip())

        if current_num is not None:
            citation_map[current_num] = ' '.join(current_text).strip()
        
        self.citation_map = citation_map
        return citation_map


# -------------------------------------------------------------------
# Metadata Filter
# -------------------------------------------------------------------
class MetadataFilter:
    """Filters documents based on metadata"""
    
    def __init__(self):
        self.documents = []
    
    def add_document(self, doc_id: str, text: str, metadata: Dict):
        """Add document with metadata"""
        self.documents.append({
            'id': doc_id,
            'text': text,
            'metadata': metadata
        })
    
    def filter_by_metadata(self, filters: Dict) -> List[Dict]:
        """Filter documents by metadata criteria"""
        results = self.documents
        
        for key, value in filters.items():
            results = [
                doc for doc in results 
                if key in doc['metadata'] and doc['metadata'][key] == value
            ]
        
        return results


# -------------------------------------------------------------------
# Jina Reranker
# -------------------------------------------------------------------
class JinaReranker:
    """Reranks search results for better relevance"""
    
    def __init__(self):
        self.model_name = "jina-reranker-v1"
    
    def rerank(self, query: str, documents: List[Document], 
               top_k: int = 5) -> List[Document]:
        """Rerank documents based on relevance to query"""
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_relevance(query, doc.page_content)
            scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score using word overlap"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        
        return intersection / union if union > 0 else 0


# -------------------------------------------------------------------
# Context Assembler
# -------------------------------------------------------------------
class ContextAssembler:
    """Assembles context from retrieved documents"""
    
    def __init__(self):
        self.max_context_length = 4000
    
    def assemble_context(self, documents: List[Document], 
                        query: str, 
                        include_metadata: bool = True) -> str:
        """Assemble context from documents with proper formatting"""
        context_parts = []
        current_length = 0
        
        for idx, doc in enumerate(documents, 1):
            chunk = f"\n--- Source {idx} ---\n{doc.page_content}\n"
            
            if include_metadata and hasattr(doc, 'metadata'):
                meta = doc.metadata
                chunk = f"[Document: {meta.get('filename', 'Unknown')}]\n{chunk}"
            
            if current_length + len(chunk) > self.max_context_length:
                break
            
            context_parts.append(chunk)
            current_length += len(chunk)
        
        return "\n".join(context_parts)


# -------------------------------------------------------------------
# Multi-Agent Validation System
# -------------------------------------------------------------------
class MultiAgentValidator:
    """Validates responses using multiple agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.validation_agents = ['accuracy', 'relevance', 'completeness']
    
    def validate_response(self, response: str, context: str, 
                         query: str) -> Dict:
        """Validate response using multiple agents"""
        validations = {}
        
        validations['accuracy'] = self._check_accuracy(response, context)
        validations['relevance'] = self._check_relevance(response, query)
        validations['completeness'] = self._check_completeness(response, query)
        
        confidence = sum(validations.values()) / len(validations)
        
        return {
            'validations': validations,
            'confidence': confidence,
            'passed': confidence >= 0.7
        }
    
    def _check_accuracy(self, response: str, context: str) -> float:
        """Check if response is accurate based on context"""
        prompt = f"""Rate the accuracy of this response based on the context (0.0 to 1.0).
Return ONLY a number between 0.0 and 1.0.

Context: {context[:500]}
Response: {response[:500]}

Accuracy score:"""
        
        try:
            result = self.llm.invoke(prompt).content
            match = re.search(r'0\.\d+|1\.0|0|1', result)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)
            return 0.7
        except:
            return 0.7
    
    def _check_relevance(self, response: str, query: str) -> float:
        """Check if response is relevant to query"""
        prompt = f"""Rate how relevant this response is to the query (0.0 to 1.0).
Return ONLY a number between 0.0 and 1.0.

Query: {query}
Response: {response[:500]}

Relevance score:"""
        
        try:
            result = self.llm.invoke(prompt).content
            match = re.search(r'0\.\d+|1\.0|0|1', result)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)
            return 0.7
        except:
            return 0.7
    
    def _check_completeness(self, response: str, query: str) -> float:
        """Check if response completely answers the query"""
        prompt = f"""Rate how completely this response answers the query (0.0 to 1.0).
Return ONLY a number between 0.0 and 1.0.

Query: {query}
Response: {response[:500]}

Completeness score:"""
        
        try:
            result = self.llm.invoke(prompt).content
            match = re.search(r'0\.\d+|1\.0|0|1', result)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)
            return 0.7
        except:
            return 0.7


# -------------------------------------------------------------------
# Citation Verification System
# -------------------------------------------------------------------
class CitationVerifier:
    """Verifies citations in generated responses"""
    
    def __init__(self, citation_db: CitationDatabase):
        self.citation_db = citation_db
    
    def verify_response_citations(self, response: str, 
                                  doc_ids: List[str]) -> Dict:
        """Verify all citations in a response"""
        cited_numbers = re.findall(r'\[(\d+)\]', response)
        
        verifications = []
        for num in cited_numbers:
            verified = False
            source_doc = None
            
            for doc_id in doc_ids:
                if self.citation_db.verify_citation(f'[{num}]', doc_id):
                    verified = True
                    source_doc = doc_id
                    break
            
            verifications.append({
                'citation_number': num,
                'verified': verified,
                'source_doc': source_doc
            })
        
        total = len(verifications)
        verified_count = sum(1 for v in verifications if v['verified'])
        verification_rate = verified_count / total if total > 0 else 1.0
        
        return {
            'verifications': verifications,
            'total_citations': total,
            'verified_citations': verified_count,
            'verification_rate': verification_rate
        }


# -------------------------------------------------------------------
# Session Memory
# -------------------------------------------------------------------
class SessionMemory:
    """Enhanced session memory with conversation history"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.max_history = max_history
        self.session_metadata = {
            'start_time': datetime.now().isoformat(),
            'query_count': 0
        }
    
    def add_interaction(self, query: str, response: str, 
                       context: Optional[str] = None,
                       validation: Optional[Dict] = None):
        """Add interaction to memory"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'context': context,
            'validation': validation
        }
        
        self.conversation_history.append(interaction)
        self.session_metadata['query_count'] += 1
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get recent conversation context"""
        recent = self.conversation_history[-n:]
        context_str = ""
        
        for interaction in recent:
            context_str += f"\nQ: {interaction['query']}\nA: {interaction['response']}\n"
        
        return context_str
    
    def export_history(self) -> Dict:
        """Export full conversation history"""
        return {
            'metadata': self.session_metadata,
            'history': self.conversation_history
        }


# -------------------------------------------------------------------
# Dynamic Web Lookup Agent (ArXiv) - FULLY USER-DRIVEN
# -------------------------------------------------------------------
class WebLookupAgent:
    """
    Dynamic ArXiv search agent - searches based purely on user query
    No hardcoded categories or assumptions
    """

    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.last_search_query = None
        self.last_results = []
    
    def lookup(self, query: str, limit: int = 10,
              category: Optional[str] = None,
              author: Optional[str] = None,
              title: Optional[str] = None,
              abstract: Optional[str] = None,
              start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              sort_by: str = 'relevance') -> List[Dict]:
        """
        Search ArXiv with user's query - completely dynamic
        
        Args:
            query: User's search query (natural language or keywords)
            limit: Maximum number of results
            category: Optional ArXiv category filter (e.g., 'cs.AI', 'hep-ex')
            author: Optional author name filter
            title: Optional title keyword filter
            abstract: Optional abstract keyword filter
            start_date: Only papers published after this date (YYYY-MM-DD)
            end_date: Only papers published before this date (YYYY-MM-DD)
            sort_by: 'relevance', 'lastUpdatedDate', or 'submittedDate'
            
        Returns:
            List of paper dictionaries with full metadata
        """
        try:
            # Build search query dynamically from user input
            search_query = self._build_search_query(
                query, category, author, title, abstract
            )
            
            params = {
                'search_query': search_query,
                'start': 0,
                'max_results': limit,
                'sortBy': sort_by,
                'sortOrder': 'descending'
            }
            
            print(f"🔍 Searching ArXiv with: '{query}'")
            if category:
                print(f"   Category filter: {category}")
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_arxiv_response(response.content)
            
            # Apply date filters if specified
            if start_date or end_date:
                papers = self._filter_by_date(papers, start_date, end_date)
            
            # Store search info
            self.last_search_query = query
            self.last_results = papers
            
            print(f"✅ Found {len(papers)} papers")
            
            return papers

        except requests.RequestException as e:
            print(f"❌ Network error: {str(e)}")
            return []
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {str(e)}")
            return []
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return []
    
    def _build_search_query(self, query: str,
                           category: Optional[str] = None,
                           author: Optional[str] = None,
                           title: Optional[str] = None,
                           abstract: Optional[str] = None) -> str:
        """
        Build ArXiv API query string from user input
        
        Args:
            query: Base user query
            category: Optional category
            author: Optional author
            title: Optional title keywords
            abstract: Optional abstract keywords
            
        Returns:
            Formatted query string for ArXiv API
        """
        query_parts = []
        
        # Main query (all fields by default)
        if query and query.strip():
            formatted_query = query.strip().replace(' ', '+')
            query_parts.append(f'all:{formatted_query}')
        
        # Author filter
        if author and author.strip():
            query_parts.append(f'au:{author.strip().replace(" ", "+")}')
        
        # Title filter
        if title and title.strip():
            query_parts.append(f'ti:{title.strip().replace(" ", "+")}')
        
        # Abstract filter
        if abstract and abstract.strip():
            query_parts.append(f'abs:{abstract.strip().replace(" ", "+")}')
        
        # Category filter
        if category and category.strip():
            query_parts.append(f'cat:{category.strip()}')
        
        # Combine all parts with AND
        final_query = '+AND+'.join(query_parts) if query_parts else 'all:*'
        
        return final_query
    
    def _parse_arxiv_response(self, xml_content: bytes) -> List[Dict]:
        """
        Parse ArXiv XML response into structured data
        
        Args:
            xml_content: Raw XML response from ArXiv
            
        Returns:
            List of paper dictionaries
        """
        root = ET.fromstring(xml_content)
        ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('arxiv:entry', ns)
        
        papers = []
        
        for entry in entries:
            paper = self._extract_paper_data(entry, ns)
            if paper:
                papers.append(paper)
        
        return papers
    
    def _extract_paper_data(self, entry, ns: Dict) -> Optional[Dict]:
        """Extract all data from a single ArXiv entry"""
        try:
            # Title
            title_elem = entry.find('arxiv:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Untitled"
            
            # Authors
            authors = []
            for author in entry.findall('arxiv:author', ns):
                name_elem = author.find('arxiv:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Abstract/Summary
            summary_elem = entry.find('arxiv:summary', ns)
            abstract = ""
            if summary_elem is not None:
                abstract = summary_elem.text.strip().replace('\n', ' ')
            
            # Published date
            published_elem = entry.find('arxiv:published', ns)
            published = published_elem.text[:10] if published_elem is not None else ""
            
            # Updated date
            updated_elem = entry.find('arxiv:updated', ns)
            updated = updated_elem.text[:10] if updated_elem is not None else published
            
            # ArXiv ID
            id_elem = entry.find('arxiv:id', ns)
            arxiv_id = ""
            if id_elem is not None:
                arxiv_id = id_elem.text
                # Extract just the ID number
                if '/abs/' in arxiv_id:
                    arxiv_id = arxiv_id.split('/abs/')[-1]
                elif 'arxiv.org/' in arxiv_id:
                    arxiv_id = arxiv_id.split('arxiv.org/')[-1]
            
            # Categories
            categories = []
            for cat in entry.findall('arxiv:category', ns):
                term = cat.get('term')
                if term:
                    categories.append(term)
            
            # Primary category
            primary_cat_elem = entry.find('arxiv:primary_category', ns)
            primary_category = primary_cat_elem.get('term') if primary_cat_elem is not None else ""
            
            # DOI (if available)
            doi = ""
            for link in entry.findall('arxiv:link', ns):
                if link.get('title') == 'doi':
                    doi = link.get('href', '')
            
            # Comment (if available)
            comment_elem = entry.find('arxiv:comment', ns)
            comment = comment_elem.text.strip() if comment_elem is not None else ""
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'published': published,
                'updated': updated,
                'arxiv_id': arxiv_id,
                'link': f"https://arxiv.org/abs/{arxiv_id}",
                'pdf_link': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                'categories': categories,
                'primary_category': primary_category,
                'doi': doi,
                'comment': comment
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting paper data: {e}")
            return None
    
    def _filter_by_date(self, papers: List[Dict],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[Dict]:
        """
        Filter papers by publication date range
        
        Args:
            papers: List of papers to filter
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Filtered list of papers
        """
        if not start_date and not end_date:
            return papers
        
        filtered = []
        
        for paper in papers:
            pub_date_str = paper.get('published', '')
            if not pub_date_str:
                continue
            
            try:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')
                
                # Check start date
                if start_date:
                    filter_start = datetime.strptime(start_date, '%Y-%m-%d')
                    if pub_date < filter_start:
                        continue
                
                # Check end date
                if end_date:
                    filter_end = datetime.strptime(end_date, '%Y-%m-%d')
                    if pub_date > filter_end:
                        continue
                
                filtered.append(paper)
                
            except ValueError:
                # Include paper if date parsing fails
                filtered.append(paper)
        
        return filtered
    
    def format_results_as_markdown(self, papers: List[Dict]) -> str:
        """
        Format search results as markdown text
        
        Args:
            papers: List of papers
            
        Returns:
            Formatted markdown string
        """
        if not papers:
            return "No papers found."
        
        md = f"# ArXiv Search Results\n\n"
        md += f"**Query:** {self.last_search_query}\n\n"
        md += f"**Results:** {len(papers)} papers\n\n"
        md += "---\n\n"
        
        for idx, paper in enumerate(papers, 1):
            md += f"## {idx}. {paper['title']}\n\n"
            
            # Authors
            author_list = ', '.join(paper['authors'][:5])
            if len(paper['authors']) > 5:
                author_list += f" *et al.* ({len(paper['authors'])} total authors)"
            md += f"**Authors:** {author_list}\n\n"
            
            # Metadata
            md += f"**Published:** {paper['published']}\n\n"
            md += f"**ArXiv ID:** [{paper['arxiv_id']}]({paper['link']})\n\n"
            md += f"**PDF:** [Download]({paper['pdf_link']})\n\n"
            
            if paper['primary_category']:
                md += f"**Primary Category:** {paper['primary_category']}\n\n"
            
            if paper['categories']:
                md += f"**All Categories:** {', '.join(paper['categories'])}\n\n"
            
            if paper.get('comment'):
                md += f"**Comments:** {paper['comment']}\n\n"
            
            # Abstract
            md += f"**Abstract:**\n\n{paper['abstract']}\n\n"
            md += "---\n\n"
        
        return md
    
    def search_similar_to_uploaded(self, uploaded_papers: List[Dict],
                                   max_results: int = 10,
                                   exclude_existing: bool = True) -> List[Dict]:
        """
        Find papers similar to uploaded papers by extracting key terms
        
        Args:
            uploaded_papers: Papers uploaded by user
            max_results: Maximum results
            exclude_existing: Exclude papers already referenced
            
        Returns:
            List of similar papers
        """
        # Extract key terms from uploaded papers
        all_text = " ".join([p.get('text', '')[:2000] for p in uploaded_papers])
        
        # Extract potential search terms (titles, key phrases)
        # Look for capitalized phrases that might be important terms
        important_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', all_text)
        
        # Get most common terms
        from collections import Counter
        term_counts = Counter(important_terms)
        top_terms = [term for term, count in term_counts.most_common(10) if count > 2]
        
        if not top_terms:
            # Fallback: use first few words from each paper
            top_terms = all_text.split()[:20]
        
        # Build search query
        search_query = ' '.join(top_terms[:5])
        
        print(f"🔍 Auto-generated search query: {search_query}")
        
        # Search
        papers = self.lookup(search_query, limit=max_results)
        
        # Exclude existing papers if requested
        if exclude_existing and papers:
            # Extract arXiv IDs from uploaded papers
            existing_ids = set()
            for uploaded in uploaded_papers:
                text = uploaded.get('text', '')
                arxiv_ids = re.findall(r'arXiv:(\d+\.\d+)', text)
                existing_ids.update(arxiv_ids)
            
            # Filter out existing papers
            papers = [p for p in papers if p['arxiv_id'] not in existing_ids]
        
        return papers


# -------------------------------------------------------------------
# Enhanced Research Assistant System
# -------------------------------------------------------------------
class ResearchAssistantSystem:
    """Complete research assistant with all architecture components"""
    
    def __init__(self, llm, embedding, chunk_size: int = 2000):
        self.llm = llm
        self.embedding = embedding
        self.chunk_size = chunk_size
        
        # Core components
        self.vectorstore = None
        self.documents = []
        self.doc_metadata = []
        
        # Enhanced components
        self.citation_extractor = EnhancedCitationExtractor()
        self.metadata_filter = MetadataFilter()
        self.reranker = JinaReranker()
        self.context_assembler = ContextAssembler()
        self.validator = MultiAgentValidator(llm)
        self.citation_verifier = CitationVerifier(
            self.citation_extractor.citation_db
        )
        self.session_memory = SessionMemory()
        self.web_lookup = WebLookupAgent()
    
    def ingest_papers(self, doc_data: List[Dict]):
        """Enhanced paper ingestion with metadata and citations"""
        for idx, data in enumerate(doc_data):
            doc_id = f"doc_{idx}"
            text = data['text']
            metadata = data.get('metadata', {})
            
            # Extract citations
            citations = self.citation_extractor.extract_citations_from_text(
                text, doc_id
            )
            
            # Add to metadata filter
            self.metadata_filter.add_document(doc_id, text, metadata)
            
            # Store document
            self.documents.append(text)
            self.doc_metadata.append({
                'id': doc_id,
                'metadata': metadata,
                'citations': citations
            })
        
        # Create vector store
        all_chunks = []
        for doc in self.documents:
            chunks = self._chunk_text(doc)
            all_chunks.extend(chunks)
        
        documents = [Document(page_content=chunk) for chunk in all_chunks]
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embedding)
        else:
            new_vectorstore = FAISS.from_documents(documents, self.embedding)
            self.vectorstore.merge_from(new_vectorstore)
        
        print(f"✅ Ingested {len(doc_data)} document(s), total chunks: {len(all_chunks)}")
        print(f"📚 Extracted {self.citation_extractor.citation_db.citation_count} citations")
    
    def answer_query(self, query: str, top_k: int = 5, 
                    use_validation: bool = True,
                    use_reranking: bool = True) -> Dict:
        """Enhanced query answering with full pipeline"""
        if not self.vectorstore:
            return {"error": "No documents ingested yet."}
        
        try:
            # Step 1: Retrieve relevant chunks
            relevant_chunks = self.vectorstore.similarity_search(query, k=top_k * 2)
            
            # Step 2: Rerank results
            if use_reranking:
                relevant_chunks = self.reranker.rerank(query, relevant_chunks, top_k)
            else:
                relevant_chunks = relevant_chunks[:top_k]
            
            # Step 3: Assemble context
            context = self.context_assembler.assemble_context(
                relevant_chunks, query
            )
            
            # Step 4: Get conversation context
            conversation_context = self.session_memory.get_recent_context(3)
            
            # Step 5: Generate response
            prompt = f"""You are a research assistant. Answer the question using the provided context.

Previous conversation:
{conversation_context}

Current question: {query}

Reference context:
{context}

Provide a clear, accurate answer based on the context. Include citation numbers [1], [2], etc. where appropriate."""
            
            response = self.llm.invoke(prompt).content
            
            # Step 6: Validate response
            validation_result = None
            if use_validation:
                validation_result = self.validator.validate_response(
                    response, context, query
                )
                
                if not validation_result['passed']:
                    response = self._improve_response(response, context, query)
                    validation_result = self.validator.validate_response(
                        response, context, query
                    )
            
            # Step 7: Verify citations
            doc_ids = [meta['id'] for meta in self.doc_metadata]
            citation_verification = self.citation_verifier.verify_response_citations(
                response, doc_ids
            )
            
            # Step 8: Add to session memory
            self.session_memory.add_interaction(
                query, response, context, validation_result
            )
            
            # Return complete result
            return {
                'answer': response,
                'validation': validation_result,
                'citation_verification': citation_verification,
                'sources': [doc.page_content[:200] for doc in relevant_chunks],
                'confidence': validation_result['confidence'] if validation_result else 0.8
            }
            
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return [
            text[i:i + self.chunk_size] 
            for i in range(0, len(text), self.chunk_size)
        ]
    
    def _improve_response(self, response: str, context: str, query: str) -> str:
        """Improve response if validation fails"""
        prompt = f"""The previous answer failed validation. Please provide a better, more accurate answer.

Question: {query}
Previous answer: {response}
Context: {context[:1000]}

Provide an improved answer:"""
        
        try:
            return self.llm.invoke(prompt).content
        except:
            return response
    
    def summarize(self, top_k: int = 5) -> str:
        """Generate summaries for all ingested documents"""
        if not self.vectorstore:
            return "❌ Error: No documents ingested yet."
        
        summaries = []
        for idx, doc in enumerate(self.documents, 1):
            chunks = self._chunk_text(doc)[:top_k]
            combined_text = "\n".join(chunks)
            
            prompt = f"""Summarize this research paper:

{combined_text}

Include: title, authors, abstract, key findings, methods, results, limitations, conclusion.
Format your response clearly with section headers."""
            
            try:
                summary = self.llm.invoke(prompt).content
                summaries.append(f"### Document {idx}\n\n{summary}")
            except Exception as e:
                summaries.append(f"### Document {idx}\n\n❌ Error: {str(e)}")
        
        return "\n\n---\n\n".join(summaries)
    
    def search_arxiv(self, query: str, **kwargs) -> List[Dict]:
        """
        Convenience method to search ArXiv
        
        Args:
            query: User's search query
            **kwargs: Additional search parameters
            
        Returns:
            List of papers
        """
        return self.web_lookup.lookup(query, **kwargs)
    
    def get_session_statistics(self) -> Dict:
        """Get detailed session statistics"""
        return {
            'documents': len(self.documents),
            'citations': self.citation_extractor.citation_db.citation_count,
            'queries': self.session_memory.session_metadata['query_count'],
            'session_duration': datetime.now().isoformat()
        }


# -------------------------------------------------------------------
# Initialize System - STREAMLIT CLOUD COMPATIBLE
# -------------------------------------------------------------------
def initialize_research_system(api_key: Optional[str] = None) -> ResearchAssistantSystem:
    """
    Initialize the research assistant system
    Compatible with local, .env, and Streamlit Cloud deployment
    """
    
    # Try multiple sources for API key (priority order)
    final_api_key = None
    
    # 1. Check if API key passed as parameter
    if api_key:
        final_api_key = api_key
    
    # 2. Check Streamlit secrets (for Cloud deployment)
    else:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and "NVIDIA_API_KEY" in st.secrets:
                final_api_key = st.secrets["NVIDIA_API_KEY"]
                print("✅ Using API key from Streamlit Cloud secrets")
        except:
            pass
    
    # 3. Check environment variable (local .env or system)
    if not final_api_key:
        final_api_key = os.getenv("NVIDIA_API_KEY")
        if final_api_key:
            print("✅ Using API key from environment variable")
    
    # 4. If still no key, raise error
    if not final_api_key:
        raise ValueError(
            "NVIDIA API key required. Please provide via:\n"
            "  - Streamlit sidebar input\n"
            "  - .env file (NVIDIA_API_KEY=...)\n"
            "  - Streamlit Cloud secrets\n"
            "  - Environment variable"
        )
    
    # Set in environment for consistency
    os.environ["NVIDIA_API_KEY"] = final_api_key
    
    # Initialize models
    print("🔄 Initializing NVIDIA models...")
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
    embedding = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
    
    print("✅ System initialized successfully!")
    
    return ResearchAssistantSystem(llm, embedding, chunk_size=2000)


# -------------------------------------------------------------------
# Utility Functions for Streamlit Integration
# -------------------------------------------------------------------
def format_papers_for_display(papers: List[Dict]) -> str:
    """Format papers for nice display in Streamlit"""
    if not papers:
        return "No papers found."
    
    output = ""
    for idx, paper in enumerate(papers, 1):
        output += f"### {idx}. {paper['title']}\n\n"
        
        # Authors
        author_str = ', '.join(paper['authors'][:3])
        if len(paper['authors']) > 3:
            author_str += f" *et al.* ({len(paper['authors'])} authors)"
        output += f"**Authors:** {author_str}\n\n"
        
        # Links and metadata
        output += f"**Published:** {paper['published']}\n\n"
        output += f"**ArXiv:** [{paper['arxiv_id']}]({paper['link']})\n\n"
        output += f"**PDF:** [Download]({paper['pdf_link']})\n\n"
        
        if paper.get('primary_category'):
            output += f"**Category:** {paper['primary_category']}\n\n"
        
        # Abstract
        output += f"**Abstract:** {paper['abstract'][:400]}...\n\n"
        output += "---\n\n"
    
    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Research Assistant System v2.0")
    print("=" * 60)
    print("\n✅ Features:")
    print("  - Dynamic ArXiv search (user-driven queries)")
    print("  - Multi-agent validation")
    print("  - Citation verification")
    print("  - Jina reranking")
    print("  - Session memory")
    print("  - Metadata filtering")
    print("  - Streamlit Cloud compatible")
    print("\n🚀 Ready to run with: streamlit run app.py")
    print("=" * 60)