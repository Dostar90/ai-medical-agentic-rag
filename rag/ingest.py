
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_vectors

DATA_PATH = "data/medical_papers"


def load_documents():
    documents = []
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def main():
    print("Loading documents...")
    docs = load_documents()
    
    print(f"Loaded {len(docs)} pages")
    
    print("Splitting documents...")
    chunks = split_documents(docs)
    
    print(f"Created {len(chunks)} chunks")
    
    print("Creating vector store...")
    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)
    
    print("✅ Ingestion complete!")


if __name__ == "__main__":
    main()
