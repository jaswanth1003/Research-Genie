"""
Research Genie - Streamlit Interface
AI-Powered Research Paper Analysis with Dynamic ArXiv Search
"""

import streamlit as st
import os
from typing import List, Dict
import json
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import research assistant components
from research_assistant import (
    ResearchAssistantSystem,
    read_pdf,
    WebLookupAgent,
    initialize_research_system,
    format_papers_for_display
)

# Page Configuration
st.set_page_config(
    page_title="Research Genie - AI Research Assistant",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .badge-success {
        background-color: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .badge-info {
        background-color: #2196F3;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .info-box {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    
    .success-box {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    
    .citation-verified {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .citation-unverified {
        background-color: #F44336;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .paper-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'system': None,
        'documents_ingested': False,
        'pdf_text': [],
        'summary': None,
        'chat_history': [],
        'arxiv_results': None,
        'use_reranking': True,
        'use_validation': True,
        'session_stats': None,
        'search_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Helper Functions
def initialize_system() -> bool:
    """Initialize the research assistant system"""
    if st.session_state.system is None:
        try:
            # Check for API key from multiple sources (Streamlit Cloud compatible)
            nvidia_api_key = None
            
            # Priority 1: Streamlit secrets (Cloud deployment) - SAFE CHECK
            try:
                if "NVIDIA_API_KEY" in st.secrets:
                    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
            except (FileNotFoundError, AttributeError, KeyError):
                # secrets.toml doesn't exist - that's fine
                pass
            
            # Priority 2: Environment variable (local .env)
            if not nvidia_api_key:
                nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            
            # Priority 3: Session state (from sidebar input)
            if not nvidia_api_key:
                nvidia_api_key = st.session_state.get('api_key')
            
            if not nvidia_api_key:
                return False
            
            with st.spinner("🔄 Initializing AI System..."):
                st.session_state.system = initialize_research_system(nvidia_api_key)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {str(e)}")
            return False
    
    return True


def process_uploaded_files(uploaded_files) -> bool:
    """Process uploaded PDF files"""
    try:
        doc_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"📖 Processing {uploaded_file.name}...")
            
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            pdf_data = read_pdf(temp_path)
            doc_data.append(pdf_data)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        status_text.text("🔍 Indexing documents...")
        st.session_state.system.ingest_papers(doc_data)
        st.session_state.documents_ingested = True
        st.session_state.pdf_text = doc_data
        
        st.session_state.session_stats = st.session_state.system.get_session_statistics()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        return False


def display_confidence_bar(confidence: float, label: str = "Confidence"):
    """Display a visual confidence bar"""
    percentage = int(confidence * 100)
    color = "#4CAF50" if confidence >= 0.7 else "#FF9800" if confidence >= 0.5 else "#F44336"
    
    st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600;">{label}</span>
                <span style="font-weight: bold; color: {color};">{percentage}%</span>
            </div>
            <div style="background-color: #E0E0E0; border-radius: 10px; height: 20px;">
                <div style="width: {percentage}%; height: 100%; background: {color}; border-radius: 10px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_validation_results(validation: Dict):
    """Display validation results"""
    if not validation:
        return
    
    st.markdown("### 🔍 Multi-Agent Validation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = validation['validations']['accuracy']
        st.metric("Accuracy", f"{accuracy:.2%}")
    
    with col2:
        relevance = validation['validations']['relevance']
        st.metric("Relevance", f"{relevance:.2%}")
    
    with col3:
        completeness = validation['validations']['completeness']
        st.metric("Completeness", f"{completeness:.2%}")
    
    with col4:
        confidence = validation['confidence']
        st.metric("Confidence", f"{confidence:.2%}")
    
    display_confidence_bar(confidence, "Overall Confidence")


def display_citation_verification(verification: Dict):
    """Display citation verification results"""
    if not verification:
        return
    
    st.markdown("### 📚 Citation Verification")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Citations", verification['total_citations'])
    
    with col2:
        st.metric("Verified", verification['verified_citations'])
    
    with col3:
        unverified = verification['total_citations'] - verification['verified_citations']
        st.metric("Unverified", unverified)
    
    if verification['verifications']:
        with st.expander("View Citation Details"):
            for v in verification['verifications']:
                status_badge = (
                    '<span class="citation-verified">✓ Verified</span>' 
                    if v['verified'] 
                    else '<span class="citation-unverified">✗ Unverified</span>'
                )
                st.markdown(f"**Citation [{v['citation_number']}]** {status_badge}", 
                          unsafe_allow_html=True)


def display_arxiv_paper(paper: Dict, index: int):
    """Display a single ArXiv paper nicely"""
    with st.expander(f"📄 {index}. {paper['title']}", expanded=(index <= 3)):
        # Authors
        author_display = ', '.join(paper['authors'][:5])
        if len(paper['authors']) > 5:
            author_display += f" *et al.* ({len(paper['authors'])} total authors)"
        
        st.markdown(f"**Authors:** {author_display}")
        
        # Metadata row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Published:** {paper['published']}")
        
        with col2:
            if paper.get('updated'):
                st.markdown(f"**Updated:** {paper['updated']}")
        
        with col3:
            if paper.get('primary_category'):
                st.markdown(f"**Category:** `{paper['primary_category']}`")
        
        # ArXiv ID and links
        st.markdown(f"**ArXiv ID:** `{paper['arxiv_id']}`")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"[📖 View on ArXiv]({paper['link']})")
        with col2:
            st.markdown(f"[📥 Download PDF]({paper['pdf_link']})")
        
        # Categories
        if paper.get('categories'):
            st.markdown(f"**All Categories:** {', '.join(paper['categories'])}")
        
        # Comment
        if paper.get('comment'):
            st.info(f"💬 {paper['comment']}")
        
        # Abstract
        st.markdown("**Abstract:**")
        st.markdown(f"_{paper['abstract']}_")


def create_analytics_chart():
    """Create analytics chart if plotly available"""
    if not PLOTLY_AVAILABLE or not st.session_state.chat_history:
        return None
    
    confidences = [
        chat.get('confidence', 0.8) 
        for chat in st.session_state.chat_history
    ]
    
    if not confidences:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(confidences) + 1)),
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Response Confidence Over Time",
        xaxis_title="Query Number",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    
    return fig


# Main Application
def main():
    """Main application"""
    
    # Header - UPDATED TO RESEARCH GENIE
    st.markdown('<h1 class="main-header">🧞 Research Genie</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="badge-success">Multi-Agent Validation</span>
            <span class="badge-info">Citation Verification</span>
            <span class="badge-success">Jina Reranker</span>
            <span class="badge-info">Dynamic ArXiv Search</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Welcome - UPDATED TO RESEARCH GENIE
    st.markdown("""
        <div class="info-box">
        <b>🧞 Your AI-Powered Research Genie</b><br><br>
        <b>Magical Features:</b>
        <ul>
            <li>🔍 Multi-Agent Validation for accuracy, relevance, and completeness</li>
            <li>📚 Automatic citation verification and tracking</li>
            <li>🎯 Intelligent result reranking for better relevance</li>
            <li>🌐 Dynamic ArXiv search - searches exactly what YOU want</li>
            <li>📊 Real-time performance metrics and analytics</li>
            <li>🧠 Context-aware conversation memory</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - COMPLETELY FIXED
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Safely check for API key from multiple sources - WRAPPED IN TRY-EXCEPT
        has_api_key = False
        api_key_source = None
        
        # Try Streamlit secrets first (Cloud deployment) - SAFE VERSION
        try:
            if hasattr(st, 'secrets') and "NVIDIA_API_KEY" in st.secrets:
                has_api_key = True
                api_key_source = "Streamlit Cloud"
        except:
            # Any error checking secrets - just pass
            pass
        
        # Try environment variable (.env file)
        if not has_api_key:
            env_key = os.getenv("NVIDIA_API_KEY")
            if env_key:
                has_api_key = True
                api_key_source = "Environment (.env)"
        
        # Try session state (user input from sidebar)
        if not has_api_key and st.session_state.get('api_key'):
            has_api_key = True
            api_key_source = "User Input"
        
        # Display status or request input
        if has_api_key:
            st.success(f"✅ API Key: {api_key_source}")
        else:
            st.markdown("### 🔑 API Key Required")
            api_key = st.text_input(
                "NVIDIA API Key",
                type="password",
                key="api_key",
                help="Get your free key from build.nvidia.com"
            )
            
            if not api_key:
                st.warning("⚠️ Please enter your API key")
                st.markdown("""
                    **Get your free API key:**
                    1. Visit [build.nvidia.com](https://build.nvidia.com)
                    2. Sign up/Login
                    3. Generate API key
                """)
                st.stop()
        
        st.markdown("---")
        st.markdown("## 📊 System Status")
        
        if st.session_state.documents_ingested and st.session_state.system:
            stats = st.session_state.system.get_session_statistics()
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['documents']}</div>
                    <div class="metric-label">Documents</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="metric-value">{stats['citations']}</div>
                    <div class="metric-label">Citations</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Genie Active")
        else:
            st.warning("⏳ No documents loaded")
        
        st.markdown("---")
        st.markdown("## 🛠️ Settings")
        
        st.session_state.use_reranking = st.checkbox(
            "Enable Reranking", value=True,
            help="Rerank search results for better relevance"
        )
        
        st.session_state.use_validation = st.checkbox(
            "Enable Validation", value=True,
            help="Validate responses with multi-agent system"
        )
        
        top_k = st.slider("Top K Results", 1, 10, 5,
                         help="Number of relevant chunks to retrieve")
        
        st.markdown("---")
        
        if st.button("🔄 Reset System", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Initialize
    if not initialize_system():
        st.error("❌ Failed to initialize system")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload",
        "📝 Summarize",
        "💬 Q&A",
        "🔍 ArXiv Search",
        "📊 Analytics"
    ])
    
    # Tab 1: Upload
    with tab1:
        st.markdown("### Upload Research Papers")
        
        st.info("📁 Upload PDF files and let the Genie analyze them with AI magic! ✨")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more research papers in PDF format"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 Process Documents", use_container_width=True):
                success = process_uploaded_files(uploaded_files)
                if success:
                    st.success("✅ Documents processed successfully!")
                    st.balloons()
                    st.rerun()
    
    # Tab 2: Summarize
    with tab2:
        st.markdown("### Document Summarization")
        
        if not st.session_state.documents_ingested:
            st.warning("⚠️ Please upload documents first in the Upload tab")
        else:
            st.info(f"📚 {len(st.session_state.pdf_text)} document(s) ready for Genie's analysis")
            
            if st.button("🧞 Generate Summary", use_container_width=True):
                with st.spinner("✨ Genie is reading and summarizing your papers..."):
                    summary = st.session_state.system.summarize(top_k=top_k)
                    st.session_state.summary = summary
            
            if st.session_state.summary:
                st.markdown("---")
                st.markdown(st.session_state.summary)
                
                st.markdown("---")
                st.download_button(
                    "💾 Download Summary",
                    data=st.session_state.summary,
                    file_name=f"research_genie_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Tab 3: Q&A
    with tab3:
        st.markdown("### Interactive Q&A")
        
        if not st.session_state.documents_ingested:
            st.warning("⚠️ Please upload documents first in the Upload tab")
        else:
            st.info("💬 Ask the Genie anything about your research papers! 🧞")
            
            # Display history
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                
                with st.chat_message("assistant", avatar="🧞"):
                    st.write(chat["answer"])
                    
                    if 'validation' in chat and chat['validation']:
                        with st.expander("View Quality Metrics"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                display_validation_results(chat['validation'])
                            
                            with col2:
                                if 'citation_verification' in chat:
                                    display_citation_verification(chat['citation_verification'])
            
            # Chat input
            query = st.chat_input("Ask the Genie a question...")
            
            if query:
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant", avatar="🧞"):
                    with st.spinner("🧞 Genie is thinking..."):
                        result = st.session_state.system.answer_query(
                            query,
                            top_k=top_k,
                            use_validation=st.session_state.use_validation,
                            use_reranking=st.session_state.use_reranking
                        )
                        
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.write(result['answer'])
                            
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if result.get('validation'):
                                    display_validation_results(result['validation'])
                            
                            with col2:
                                if result.get('citation_verification'):
                                    display_citation_verification(result['citation_verification'])
                            
                            st.session_state.chat_history.append({
                                "question": query,
                                "answer": result['answer'],
                                "validation": result.get('validation'),
                                "citation_verification": result.get('citation_verification'),
                                "confidence": result.get('confidence', 0.8),
                                "timestamp": datetime.now().isoformat()
                            })
            
            if st.session_state.chat_history:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🗑️ Clear Chat History", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
    
    # Tab 4: Enhanced Dynamic ArXiv Search
    with tab4:
        st.markdown("### 🔍 Dynamic ArXiv Literature Search")
        
        st.markdown("""
            Search ArXiv with **your own queries** - the Genie searches exactly what you ask for! 🧞✨
        """)
        
        # Search type selector
        search_mode = st.radio(
            "Search Mode:",
            ["Simple Search", "Advanced Search", "Find Similar Papers"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # ========== SIMPLE SEARCH ==========
        if search_mode == "Simple Search":
            st.markdown("### 📝 Simple Search")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_query = st.text_input(
                    "What would you like to search for?",
                    placeholder="e.g., quantum computing, machine learning, neural networks",
                    help="Enter any topic you're interested in"
                )
            
            with col2:
                max_results = st.number_input("Max Results", 5, 50, 10, step=5)
            
            if st.button("🔍 Search ArXiv", use_container_width=True, key="simple_search"):
                if user_query and user_query.strip():
                    with st.spinner(f"🧞 Genie is searching ArXiv for: '{user_query}'..."):
                        try:
                            agent = WebLookupAgent()
                            papers = agent.lookup(user_query, limit=max_results)
                            
                            st.session_state.arxiv_results = papers
                            
                            # Add to search history
                            st.session_state.search_history.append({
                                'query': user_query,
                                'results': len(papers),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            if papers:
                                st.success(f"✨ Genie found {len(papers)} papers!")
                            else:
                                st.warning("No papers found. Try different keywords.")
                                
                        except Exception as e:
                            st.error(f"❌ Search error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a search query")
        
        # ========== ADVANCED SEARCH ==========
        elif search_mode == "Advanced Search":
            st.markdown("### ⚙️ Advanced Search with Filters")
            
            # Main query
            user_query = st.text_input(
                "Search Query (required)",
                placeholder="Your research topic or keywords",
                help="Main search terms"
            )
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Filters:**")
                
                author_filter = st.text_input(
                    "Author Name (optional)",
                    placeholder="e.g., Einstein, Feynman"
                )
                
                title_filter = st.text_input(
                    "Title Keywords (optional)",
                    placeholder="Keywords that must appear in title"
                )
                
                abstract_filter = st.text_input(
                    "Abstract Keywords (optional)",
                    placeholder="Keywords that must appear in abstract"
                )
            
            with col2:
                st.markdown("**Metadata Filters:**")
                
                category_filter = st.text_input(
                    "ArXiv Category (optional)",
                    placeholder="e.g., cs.AI, hep-ex, physics.comp-ph",
                    help="Specific ArXiv category code"
                )
                
                sort_by = st.selectbox(
                    "Sort Results By",
                    ["relevance", "lastUpdatedDate", "submittedDate"]
                )
                
                time_filter = st.selectbox(
                    "Time Period",
                    ["All time", "Last year", "Last 2 years", "Last 5 years", "Custom range"]
                )
                
                # Custom date range
                if time_filter == "Custom range":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        start_date = st.date_input("From", datetime.now() - timedelta(days=365))
                    with col_b:
                        end_date = st.date_input("To", datetime.now())
            
            # Max results
            max_results = st.slider("Maximum Results", 5, 50, 15, step=5)
            
            # Search button
            if st.button("🔍 Advanced Search", use_container_width=True, key="adv_search"):
                if user_query and user_query.strip():
                    with st.spinner("🧞 Genie is searching with your filters..."):
                        try:
                            agent = WebLookupAgent()
                            
                            # Determine date filters
                            start_date_str = None
                            end_date_str = None
                            
                            if time_filter == "Last year":
                                start_date_str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                            elif time_filter == "Last 2 years":
                                start_date_str = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                            elif time_filter == "Last 5 years":
                                start_date_str = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
                            elif time_filter == "Custom range":
                                start_date_str = start_date.strftime('%Y-%m-%d')
                                end_date_str = end_date.strftime('%Y-%m-%d')
                            
                            # Perform advanced search
                            papers = agent.lookup(
                                query=user_query,
                                limit=max_results,
                                category=category_filter if category_filter.strip() else None,
                                author=author_filter if author_filter.strip() else None,
                                title=title_filter if title_filter.strip() else None,
                                abstract=abstract_filter if abstract_filter.strip() else None,
                                start_date=start_date_str,
                                end_date=end_date_str,
                                sort_by=sort_by
                            )
                            
                            st.session_state.arxiv_results = papers
                            
                            if papers:
                                st.success(f"✨ Genie found {len(papers)} papers!")
                            else:
                                st.warning("No papers found. Try adjusting filters.")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a search query")
        
        # ========== FIND SIMILAR PAPERS ==========
        elif search_mode == "Find Similar Papers":
            st.markdown("### 🧠 AI-Powered Similar Paper Discovery")
            
            if not st.session_state.documents_ingested:
                st.warning("⚠️ Upload documents first to use this feature")
            else:
                st.info("""
                    🧞 Let the Genie analyze your papers and magically find related research!
                """)
                
                max_results = st.slider("Maximum Results", 5, 30, 15, step=5)
                
                exclude_existing = st.checkbox(
                    "Exclude papers already referenced",
                    value=True
                )
                
                if st.button("🔍 Find Similar Papers", use_container_width=True, key="similar_search"):
                    with st.spinner("🧞 Genie is analyzing your papers..."):
                        try:
                            agent = WebLookupAgent()
                            
                            papers = agent.search_similar_to_uploaded(
                                st.session_state.pdf_text,
                                max_results=max_results,
                                exclude_existing=exclude_existing
                            )
                            
                            st.session_state.arxiv_results = papers
                            
                            if papers:
                                st.success(f"✨ Genie found {len(papers)} similar papers!")
                            else:
                                st.warning("No similar papers found.")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        # ========== DISPLAY RESULTS ==========
        if st.session_state.arxiv_results:
            st.markdown("---")
            st.markdown(f"### 📚 Search Results ({len(st.session_state.arxiv_results)} papers)")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                md_results = format_papers_for_display(st.session_state.arxiv_results)
                st.download_button(
                    "💾 Download Markdown",
                    data=md_results,
                    file_name=f"research_genie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                json_results = json.dumps(st.session_state.arxiv_results, indent=2)
                st.download_button(
                    "💾 Download JSON",
                    data=json_results,
                    file_name=f"research_genie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Display papers
            for idx, paper in enumerate(st.session_state.arxiv_results, 1):
                display_arxiv_paper(paper, idx)
        
        # Search suggestions
        elif search_mode == "Simple Search":
            st.markdown("---")
            st.markdown("### 💡 Popular Topics")
            st.markdown("✨ Click any topic and let Genie search for you:")
            
            suggestions = [
                "machine learning particle physics",
                "quantum computing algorithms",
                "gravitational wave detection",
                "natural language processing",
                "computer vision deep learning",
                "reinforcement learning",
                "neural architecture search",
                "large language models"
            ]
            
            cols = st.columns(4)
            for idx, suggestion in enumerate(suggestions):
                with cols[idx % 4]:
                    if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                        agent = WebLookupAgent()
                        papers = agent.lookup(suggestion, limit=10)
                        st.session_state.arxiv_results = papers
                        st.session_state.search_history.append({
                            'query': suggestion,
                            'results': len(papers),
                            'timestamp': datetime.now().isoformat()
                        })
                        st.rerun()
    
    # Tab 5: Analytics
    with tab5:
        st.markdown("### 📊 Session Analytics")
        
        if not st.session_state.chat_history and not st.session_state.search_history:
            st.info("💡 No analytics yet. Start using the Genie to see magic happen! 🧞✨")
        else:
            # Q&A Analytics
            if st.session_state.chat_history:
                st.markdown("#### 💬 Q&A Performance")
                
                if PLOTLY_AVAILABLE:
                    fig = create_analytics_chart()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Queries", len(st.session_state.chat_history))
                
                with col2:
                    avg_conf = sum(c.get('confidence', 0) for c in st.session_state.chat_history) / len(st.session_state.chat_history)
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                
                with col3:
                    if st.session_state.system:
                        stats = st.session_state.system.get_session_statistics()
                        st.metric("Citations", stats.get('citations', 0))
            
            # Search Analytics
            if st.session_state.search_history:
                st.markdown("---")
                st.markdown("#### 🔍 Search History")
                
                for idx, search in enumerate(reversed(st.session_state.search_history[-10:]), 1):
                    st.markdown(
                        f"{idx}. **{search['query']}** - "
                        f"{search['results']} results"
                    )
    
    # Footer - UPDATED TO RESEARCH GENIE
    st.markdown("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid #E0E0E0; color: #757575;">
            <p><b>🧞 Research Genie v2.0</b></p>
            <p>Your magical AI-powered research companion</p>
            <p>Built with ❤️ using Streamlit, LangChain & NVIDIA AI</p>
            <p>© 2024</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()