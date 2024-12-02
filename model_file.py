import os
from typing import Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
import os


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)


def load_text_from_file(file_path: str) -> Optional[str]:
    """Load text from either a PDF or text file."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    print(f"Unsupported file type for {file_path}, skipping.")
    return None


def build_vector_store(
    directory_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
) -> FAISS:
    """Build a FAISS vector store from documents in a directory."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
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

    return FAISS.from_documents(documents, embeddings)


def construct_prompt(
    system_prompt: str, retrieved_docs: str, user_query: str
) -> str:
    """Construct a prompt for the LLM based on the system prompt, retrieved docs, and query."""
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


def query_llm(
    vector_store: FAISS, query: str, temperature: float = 0
) -> str:
    """Query the LLM based on the vector store and user query."""
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
    llm = ChatOllama(model=os.environ['LLM_MODEL'], temperature=temperature)
    return llm.invoke(final_prompt)


def run_llm():
    """Main function to execute the workflow."""
    print("Starting vector store creation...")
    vector_store = build_vector_store("./Document/test")
    print("Vector store creation complete.")
    
    user_query = "How to download and install MFP agent...?"
    print("Querying the LLM...")
    response = query_llm(vector_store, user_query)
    print("LLM Response:")
    print(response)


if __name__ == "__main__":
    run_llm()