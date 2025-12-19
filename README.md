# Research Genie

An agentic AI-based research assistant for ingesting, summarizing, and querying academic papers with LLM-powered agents and embeddings.


An AI-powered Research Assistant System that helps researchers and students interact with scientific papers.
This project enables:

- PDF ingestion & text extraction
- Intelligent document chunking & FAISS vector search
- Literature scanning using NVIDIA embeddings
- Citation extraction & reference mapping
- LLM-powered structured summarization
- ArXiv API lookup for related academic papers
- Interactive Gradio UI for summarization & question answering


## **Features**

- PDF Reader – Extracts text from uploaded research papers.
- Embeddings + FAISS – Creates a searchable knowledge base of ingested papers.
- Citation Extractor – Parses reference sections to map citations.
- Summarization Agent – Generates structured summaries (title, authors, abstract, key findings, methods, results, limitations, conclusion).
- Web Lookup Agent – Searches related papers on ArXiv.
- Interactive Gradio App – Upload PDFs, get summaries, and ask questions.






## **Installation**

Clone this repository and install dependencies:

```
git clone https://github.com/your-username/research-assistant.git
cd research-assistant

# Install dependencies
pip install faiss-cpu PyPDF2 langchain-nvidia-ai-endpoints langchain-community gradio requests 
```

## **Setup**

Get your NVIDIA API key from NVIDIA AI Endpoints.

Add it to your environment:
```
export NVIDIA_API_KEY="your_api_key_here"
```

## **Usage**

Run the assistant in two ways:

1. Command Line Script

```
python research_assistant.py
```

-Ingests your uploaded PDF.
-Generates a structured summary (title, abstract, methods, results, etc.).
-Allows you to query the ingested paper.

2️.Gradio Web App
```
python research_assistant.py
```

-Opens a browser interface.
-Summarize Tab → Upload a PDF to get structured summaries.
-Ask Questions Tab → Enter natural-language questions about the paper and get answers.


## **Example Workflow**

- Upload a research paper PDF (e.g., paper.pdf).
- The system extracts text, chunks it, and stores embeddings in FAISS.
- Click Generate Summary → Get a structured summary with sections like Title, Authors, Abstract, Key Findings, Methods, Results, Limitations, Conclusion.
- Switch to Ask Questions Tab → Ask questions like:
```
What are the limitations of this paper?
what is the key role of small language models in the Agentic AI systems ?
```
- Results are displayed based only on the uploaded document (with optional ArXiv lookup for related works).

## **Tech Stack**

- Python – Core language

- LangChain – Orchestration for LLMs & embeddings

- NVIDIA LLaMA 3.1 – Large Language Model (LLM) for summarization & Q/A

- NVIDIA Embeddings – For semantic vector search

- FAISS – Vector similarity search engine

- PyPDF2 – PDF text extraction

- Gradio – Web-based user interface

- ArXiv API – Retrieve related academic papers

## **Conclusion**

The Agentic  AI Research Paper Assistant simplifies the way researchers and students interact with academic literature. By combining PDF ingestion, semantic search, citation extraction, and LLM-powered summarization, it provides an end-to-end workflow for quickly understanding and querying research papers.

Whether you want a structured summary or answers to specific questions, this tool helps you focus more on insights and less on manual reading.
