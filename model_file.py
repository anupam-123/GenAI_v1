import os
import pickle
from pypdf import PdfReader
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
import hashlib
from dotenv import load_dotenv
from utils import timer, setup_logging

load_dotenv()
logging = setup_logging()

def save_vector_store(vector_store, file_path='vector_store.pkl'):
    """Save the vector store to a file using Pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(vector_store, f)

def load_vector_store(file_path='vector_store.pkl'):
    """Load the vector store from a file using Pickle."""
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def compute_file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logging.error(f"Error computing hash for file {file_path}: {e}")
        return ""

def save_file_hash(file_path: str, hash_file_path='file_hash.txt'):
    """Save the hash of a file to a hash file."""
    file_hash = compute_file_hash(file_path)
    if file_hash:
        with open(hash_file_path, 'w') as f:
            f.write(file_hash)

def is_file_changed(file_path: str, hash_file_path='file_hash.txt') -> bool:
    """Check if the file has changed by comparing its hash with the stored hash."""
    current_hash = compute_file_hash(file_path)
    if not current_hash:
        return False

    if os.path.exists(hash_file_path):
        with open(hash_file_path, 'r') as f:
            stored_hash = f.read().strip()
        return current_hash != stored_hash
    else:
        return True

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() for page in reader.pages)
    except FileNotFoundError as e:
        logging.error(f"Error reading PDF file {pdf_path}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error while reading PDF file {pdf_path}: {e}")
        return ""

@timer
def load_text_from_file(file_path: str) -> Optional[str]:
    """Load text from either a PDF or text file."""
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        else:
            logging.warning(f"Unsupported file type for {file_path}, skipping.")
            return None
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading file {file_path}: {e}")
        return None

@timer
def build_vector_store(directory_path: str) -> FAISS:
    """Build a FAISS vector store from documents in a directory."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.environ.get("CHUNK_SIZE", 1024)),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 0))
        )
        documents = []

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                text = load_text_from_file(file_path)
                if text:
                    chunks = text_splitter.split_text(text)
                    sku = os.path.splitext(os.path.basename(file_path))[0]
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata={"file_name": sku}))
                else:
                    logging.warning(f"No text extracted from {file_path}, skipping.")
            else:
                logging.warning(f"{file_path} is not a file, skipping.")
        
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        logging.error(f"An error occurred while building the vector store: {e}")
        raise

def construct_prompt(system_prompt: str, retrieved_docs: str, user_query: str) -> str:
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(
        """
        Here is the retrieved context:
        {retrieved_docs}

        Here is the user's query:
        {user_query}
        """
    )
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    formatted_prompt = chat_prompt.format_messages(
        retrieved_docs=retrieved_docs, user_query=user_query
    )
    return "".join(message.content for message in formatted_prompt)

@timer
def query_llm(vector_store: FAISS, query: str) -> str:
    """Query the LLM based on the vector store and user query."""
    try:
        results = vector_store.similarity_search(query, k=5)
        retrieved_docs = "\n\n".join([doc.page_content for doc in results])
        system_prompt = """
        You are a highly intelligent assistant designed to answer user questions strictly based on the provided document context. Here are your responsibilities:

        1. **Use Provided Context Only**: Your answers must rely exclusively on the content retrieved from the document. Do not infer or generate information outside the document's scope.

        2. **Handle Irrelevant Queries**: If the user's query is unrelated to the provided document, respond with: 
        "The query is not relevant to the document."

        3. **Include Page Numbers**: 
        - If the answer is based on the document, include the relevant page number(s) to indicate where the information was sourced.
        - If the query is irrelevant, mention that the document context provided does not address the query and include the retrieved page numbers.

        4. **Structured Output**:
        - If the answer is based on the document, include specific details and context to ensure accuracy, along with the page number(s).
        - If the query is irrelevant, explicitly state it without additional assumptions, mentioning the retrieved page numbers.

        5. **Tone and Clarity**: Always maintain a professional and clear tone.

        6. **Content Relevance**: If you feel the retrieved document and query are relevant, answer with an appropriate and clear response, referencing the specific page(s).

        You will be provided with:
        - **Retrieved Document Context**: The content relevant to the user's query, along with page number(s).
        - **User Query**: The question or task from the user.

        Your task:
        - Analyze the retrieved document context and its page numbers.
        - Answer the query based on the document, referencing the page number(s), or identify the query as irrelevant if the document does not contain an answer.
        """
        final_prompt = construct_prompt(system_prompt, retrieved_docs, query)
        llm = ChatOllama(model=os.environ['LLM_MODEL'], temperature=os.environ['TEMPERATURE'])
        return llm.invoke(final_prompt)
    except Exception as e:
        logging.error(f"An error occurred while querying the LLM: {e}")
        return "An error occurred while processing your request."

def run_llm(user_query):
    """Main function to execute the workflow."""
    logging.info("Starting vector store creation...")
    document_directory = os.environ.get('DOCUMENT_DIRECTORY', './uploads/')
    hash_file_path = 'file_hash.txt'
    vector_store_file_path = 'vector_store.pkl'

    try:
        file_changed = any(is_file_changed(os.path.join(document_directory, filename), hash_file_path)
                           for filename in os.listdir(document_directory) if os.path.isfile(os.path.join(document_directory, filename)))

        if file_changed:
            logging.info("File has changed. Regenerating vector store...")
            vector_store = build_vector_store(document_directory)
            save_vector_store(vector_store, vector_store_file_path)
            for filename in os.listdir(document_directory):
                if os.path.isfile(os.path.join(document_directory, filename)):
                    save_file_hash(os.path.join(document_directory, filename), hash_file_path)
        else:
            logging.info("File has not changed. Loading existing vector store...")
            vector_store = load_vector_store(vector_store_file_path)

        logging.info("Querying the LLM...")
        response = query_llm(vector_store, user_query)
        logging.info("LLM Response retrieved.")
        return response
    except Exception as e:
        logging.error(f"An error occurred in run_llm: {e}")
        return "An error occurred while processing your request."