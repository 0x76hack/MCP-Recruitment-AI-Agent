import os
import logging
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
)

VECTOR_STORE = None
RESUME_DB = {}  

def initialize_rag() -> bool:
    """Initializes the FAISS vector store from PDF resumes."""
    global VECTOR_STORE, RESUME_DB
    
    resume_dir = "data/resumes"
    if not os.path.exists(resume_dir):
        os.makedirs(resume_dir)
        logger.warning(f"Created directory '{resume_dir}'. Please place PDF resumes here and restart.")
        return False

    logger.info("Loading and embedding PDF resumes. This may take a moment...")
    loader = PyPDFDirectoryLoader(resume_dir)
    docs = loader.load()

    if not docs:
        logger.warning(f"No PDFs found in '{resume_dir}'. Please add candidate resumes and restart.")
        return False

    for doc in docs:
        filename = os.path.basename(doc.metadata['source'])
        if filename not in RESUME_DB:
            RESUME_DB[filename] = ""
        RESUME_DB[filename] += doc.page_content + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        VECTOR_STORE = FAISS.from_documents(splits, embeddings)
        logger.info(f"Successfully ingested {len(RESUME_DB)} resumes into the vector database.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        return False

RAG_READY = initialize_rag()

@tool
def extract_requirements(jd: str) -> str:
    """Parses a job description to extract must-have and nice-to-have skills."""
    prompt = f"Extract 'must_have' and 'nice_to_have' skills from this JD in JSON format:\n{jd}"
    response = llm.invoke(prompt)
    return response.content

@tool
def search_resumes_tool(requirements: str) -> str:
    """Searches the vector database for resumes matching the requirements. Returns top candidates."""
    if not RAG_READY or not VECTOR_STORE:
        return "System Error: Vector database is not initialized. No resumes loaded."
    
    results = VECTOR_STORE.similarity_search(requirements, k=5)
    
    formatted_results = []
    for res in results:
        filename = os.path.basename(res.metadata['source'])
        formatted_results.append(f"Candidate File: {filename}\nRelevant Snippet: {res.page_content}\n---")
        
    return "\n".join(formatted_results)

@tool
def compare_candidates(candidate_filenames: list[str]) -> str:
    """Performs a head-to-head comparison of candidates based on their PDF filenames."""
    if not RAG_READY:
        return "System Error: Database not ready."
        
    profiles = []
    for filename in candidate_filenames:
        if filename in RESUME_DB:
            text = RESUME_DB[filename][:2500] 
            profiles.append(f"--- CANDIDATE: {filename} ---\n{text}")
        else:
            profiles.append(f"Candidate {filename} not found in the database.")
            
    if not profiles:
        return "No valid candidates found to compare."
        
    prompt = f"Compare the following candidates head-to-head based on their profiles. Highlight strengths, weaknesses, and a final recommendation for each:\n\n{chr(10).join(profiles)}"
    response = llm.invoke(prompt)
    return response.content

@tool
def generate_interview_questions(candidate_filename: str) -> str:
    """Creates custom screening questions based on a specific candidate's resume."""
    if candidate_filename not in RESUME_DB:
        return f"Candidate '{candidate_filename}' not found."
        
    text = RESUME_DB[candidate_filename][:2500]
    prompt = f"Based on this resume, identify 3 potential skill gaps or weak points. Generate 1 targeted interview question for each gap to screen the candidate effectively:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

AGENT_TOOLS = [extract_requirements, search_resumes_tool, compare_candidates, generate_interview_questions]