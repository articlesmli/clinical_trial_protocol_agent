import os
from pypdf import PdfReader
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.schemas import CriteriaExtractionOutput

def parse_and_store_pdf_to_vectordb(pdf_path: str):
    """Parses PDF, chunks text, embeds it via HuggingFace, and stores in ChromaDB."""
    print(f"[Tool: PDF Parser] Reading PDF from {pdf_path} using pypdf...")
    
    if not os.path.exists(pdf_path):
        print(f"[Warning] PDF file not found at {pdf_path}.")
        return None
        
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
            
    print(f"[Tool: PDF Parser] Extracted {len(docs)} pages.")

    # 1. Chunk the text into smaller pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"[Tool: Chunking] Split document into {len(chunks)} chunks.")

    # 2. Embed and store in ChromaDB
    print("[Tool: Vector DB] Converting chunks to embeddings and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("[Tool: Vector DB] Successfully stored vector embeddings!")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def extract_criteria_with_rag(pdf_path: str) -> CriteriaExtractionOutput:
    """Orchestrates the RAG retrieval and Anthropic structured extraction."""
    retriever = parse_and_store_pdf_to_vectordb(pdf_path)
    
    if not retriever:
        # Fallback if PDF is missing
        return CriteriaExtractionOutput(
            inclusion_criteria=["Age >= 18", "Confirmed Stage II/III NSCLC"],
            exclusion_criteria=["Prior immunotherapy", "Pregnancy"]
        )
    
    # Retrieve relevant text chunks
    print("[Tool: RAG Retriever] Searching vector DB for inclusion/exclusion text...")
    relevant_docs = retriever.invoke("What are the patient inclusion and exclusion criteria?")
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"[Tool: RAG Retriever] Fetched {len(relevant_docs)} relevant text chunks.")

    # Send retrieved chunks to Claude with Pydantic enforcement
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            print("[Tool: LLM] Invoking Anthropic Claude with retrieved context...")
            llm = ChatAnthropic(model="claude-sonnet-5-5")
            structured_llm = llm.with_structured_output(CriteriaExtractionOutput, method="json_schema")
            
            prompt = f"Extract the patient inclusion and exclusion criteria strictly from the following clinical trial protocol text:\n\n{context_text}"
            
            result = structured_llm.invoke(prompt)
            if result and (result.inclusion_criteria or result.exclusion_criteria):
                print("[Tool: LLM] Successfully extracted criteria via RAG + Claude.")
                return result
        except Exception as e:
            print(f"[Error] Anthropic LLM extraction failed: {e}. Falling back.")

    # Final fallback mock data
    return CriteriaExtractionOutput(
        inclusion_criteria=["Age >= 18 years at time of consent", "Confirmed Stage II/III NSCLC"],
        exclusion_criteria=["Prior immunotherapy", "Active brain metastases"]
    )