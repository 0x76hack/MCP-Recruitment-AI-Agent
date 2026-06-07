import os
import logging
import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("filesystem_mcp_server")

load_dotenv()

RESUME_DIR = os.getenv("RESUME_DIR", "data/resumes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    logger.error("Security Warning: OPENROUTER_API_KEY is not set in the environment variables.")

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name=OPENROUTER_MODEL
)

VECTOR_STORE = None
RESUME_DB = {}
WATCH_TASKS = {}
RAG_READY = False

mcp = FastMCP("Recruitment Filesystem MCP Server")

def initialize_rag() -> bool:
    """Initializes the FAISS vector store from PDF resumes."""
    global VECTOR_STORE, RESUME_DB, RAG_READY
    
    if not os.path.exists(RESUME_DIR):
        try:
            os.makedirs(RESUME_DIR)
            logger.warning(f"Created directory '{RESUME_DIR}'. Please place PDF resumes here.")
            return False
        except Exception as e:
            logger.error(f"Failed to create directory {RESUME_DIR}: {e}")
            return False

    logger.info(f"Loading and embedding PDF resumes from {RESUME_DIR}. This may take a moment...")
    try:
        loader = PyPDFDirectoryLoader(RESUME_DIR)
        docs = loader.load()
    except Exception as e:
        logger.error(f"Failed to load documents from {RESUME_DIR}: {e}")
        return False

    if not docs:
        logger.warning(f"No PDFs found in '{RESUME_DIR}'. Please add candidate resumes.")
        return False

    for doc in docs:
        filename = os.path.basename(doc.metadata.get('source', 'unknown'))
        if filename not in RESUME_DB:
            RESUME_DB[filename] = ""
        RESUME_DB[filename] += doc.page_content + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        VECTOR_STORE = FAISS.from_documents(splits, embeddings)
        RAG_READY = True
        logger.info(f"Successfully ingested {len(RESUME_DB)} resumes into the vector database.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        return False

initialize_rag()


@mcp.tool()
def extract_requirements(jd: str) -> str:
    """Parses a job description to extract must-have and nice-to-have skills."""
    if not jd or not jd.strip():
        return "Error: Job description text cannot be empty."
    
    try:
        prompt = f"Extract 'must_have' and 'nice_to_have' skills from this JD in JSON format:\n{jd}"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error in extract_requirements: {e}")
        return f"System Error: Failed to extract requirements. Detail: {str(e)}"

@mcp.tool()
def search_resumes_tool(requirements: str) -> str:
    """Searches the vector database for resumes matching the requirements. Returns top candidates."""
    if not RAG_READY or not VECTOR_STORE:
        return "System Error: Vector database is not initialized. No resumes loaded."
    
    if not requirements or not requirements.strip():
        return "Error: Search requirements query cannot be empty."
    
    try:
        results = VECTOR_STORE.similarity_search(requirements, k=5)
        formatted_results = []
        for res in results:
            filename = os.path.basename(res.metadata.get('source', 'unknown'))
            formatted_results.append(f"Candidate File: {filename}\nRelevant Snippet: {res.page_content}\n---")
            
        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error in search_resumes_tool: {e}")
        return f"System Error: Failed to search database. Detail: {str(e)}"

@mcp.tool()
def compare_candidates(candidate_filenames: List[str]) -> str:
    """Performs a head-to-head comparison of candidates based on their PDF filenames."""
    if not RAG_READY:
        return "System Error: Database not ready."
    
    if not candidate_filenames:
        return "Error: Candidate filenames list cannot be empty."
        
    profiles = []
    for filename in candidate_filenames:
        if filename in RESUME_DB:
            text = RESUME_DB[filename][:2500] 
            profiles.append(f"--- CANDIDATE: {filename} ---\n{text}")
        else:
            profiles.append(f"Candidate '{filename}' not found in the database.")
            
    if not profiles:
        return "No valid candidates found to compare."
        
    try:
        prompt = f"Compare the following candidates head-to-head based on their profiles. Highlight strengths, weaknesses, and a final recommendation for each:\n\n{chr(10).join(profiles)}"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error in compare_candidates: {e}")
        return f"System Error: Failed to compare candidates. Detail: {str(e)}"

@mcp.tool()
def generate_interview_questions(candidate_filename: str) -> str:
    """Creates custom screening questions based on a specific candidate's resume."""
    if not RAG_READY:
        return "System Error: Database not ready."
        
    if candidate_filename not in RESUME_DB:
        return f"Candidate '{candidate_filename}' not found in the database. Please verify name."
        
    try:
        text = RESUME_DB[candidate_filename][:2500]
        prompt = f"Based on this resume, identify 3 potential skill gaps or weak points. Generate 1 targeted interview question for each gap to screen the candidate effectively:\n\n{text}"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error in generate_interview_questions: {e}")
        return f"System Error: Failed to generate interview questions. Detail: {str(e)}"


async def watch_loop(directory_path: str):
    """Internal poll loop to check for new resumes."""
    logger.info(f"Directory watcher loop started for path: {directory_path}")
    while True:
        try:
            if os.path.exists(directory_path):
                files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
                new_files = [f for f in files if f not in RESUME_DB]
                
                if new_files:
                    logger.info(f"Watcher found {len(new_files)} new resumes. Starting batch process...")
                    full_paths = [os.path.join(directory_path, f) for f in new_files]
                    ingest_files_batch(full_paths)
            
            await asyncio.sleep(5)  
        except asyncio.CancelledError:
            logger.info(f"Watcher loop for {directory_path} has been cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in watcher loop: {e}")
            await asyncio.sleep(5)

def ingest_files_batch(file_paths: List[str]) -> Dict[str, Any]:
    """Helper to process multiple files, chunk them, and add to FAISS index."""
    global VECTOR_STORE, RESUME_DB, RAG_READY
    
    successful_files = []
    failed_files = {}
    new_documents = []
    
    for path in file_paths:
        filename = os.path.basename(path)
        if not os.path.exists(path):
            failed_files[filename] = "File not found."
            continue
            
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            if not docs:
                failed_files[filename] = "No text content extracted."
                continue
            
            content = ""
            for doc in docs:
                content += doc.page_content + "\n"
            
            RESUME_DB[filename] = content
            new_documents.extend(docs)
            successful_files.append(filename)
        except Exception as e:
            failed_files[filename] = str(e)
            
    if not new_documents:
        return {
            "processed": successful_files,
            "failed": failed_files,
            "status": "No documents to add to FAISS."
        }
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(new_documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if VECTOR_STORE is None:
            VECTOR_STORE = FAISS.from_documents(splits, embeddings)
        else:
            VECTOR_STORE.add_documents(splits)
            
        RAG_READY = True
        logger.info(f"Batch processed and added {len(successful_files)} new files to the FAISS database.")
        
        return {
            "processed": successful_files,
            "failed": failed_files,
            "status": f"Successfully indexed {len(successful_files)} candidates ({len(splits)} chunks)."
        }
    except Exception as e:
        logger.error(f"Batch FAISS index update failed: {e}")
        return {
            "processed": successful_files,
            "failed": {**failed_files, "FAISS_Index": str(e)},
            "status": "Failed to update vector store index."
        }

@mcp.tool()
def watch_directory(directory_path: str = "data/resumes") -> str:
    """Monitor for new resumes. Sets up a background monitoring loop on the directory."""
    abs_path = os.path.abspath(directory_path)
    if not os.path.exists(abs_path):
        try:
            os.makedirs(abs_path)
        except Exception as e:
            return f"Error: Directory does not exist and cannot be created: {str(e)}"
            
    if abs_path in WATCH_TASKS:
        return f"Directory '{directory_path}' is already being watched."
        
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(watch_loop(abs_path))
        WATCH_TASKS[abs_path] = task
        return f"Started watching directory: {directory_path} for new resumes."
    except Exception as e:
        logger.error(f"Failed to start watch loop: {e}")
        return f"Error: Failed to register watch loop. Detail: {str(e)}"

@mcp.tool()
def batch_process(candidate_filenames: List[str]) -> str:
    """Handle multiple files efficiently. Ingests all specified files into RAG database in a single batch."""
    if not candidate_filenames:
        return "Error: Candidate filenames list cannot be empty."
        
    full_paths = []
    for filename in candidate_filenames:
        # Prevent path traversal
        clean_name = os.path.basename(filename)
        full_paths.append(os.path.join(RESUME_DIR, clean_name))
        
    try:
        result = ingest_files_batch(full_paths)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in batch_process: {e}")
        return f"System Error: Batch processing failed. Detail: {str(e)}"


@mcp.resource("resumes://list")
def list_resumes() -> str:
    """List all candidate resumes currently indexed in the vector database."""
    resumes_list = list(RESUME_DB.keys())
    return json.dumps({
        "total_count": len(resumes_list),
        "candidates": resumes_list
    }, indent=2)

@mcp.resource("resumes://content/{filename}")
def get_resume_content(filename: str) -> str:
    """Get the raw extracted text content of a specific candidate's resume."""
    clean_name = os.path.basename(filename)
    if clean_name in RESUME_DB:
        return RESUME_DB[clean_name]
    else:
        return f"Error: Candidate resume '{clean_name}' was not found in the RAG database."

@mcp.resource("status://rag")
def get_rag_status() -> str:
    """Get the status of the FAISS vector database and index sizes."""
    return json.dumps({
        "initialized": RAG_READY,
        "resume_directory": os.path.abspath(RESUME_DIR),
        "number_of_candidates": len(RESUME_DB),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OPENROUTER_MODEL,
        "active_directory_watchers": list(WATCH_TASKS.keys())
    }, indent=2)


if __name__ == "__main__":
    logger.info("Starting Recruitment Filesystem MCP Server...")
    mcp.run()
