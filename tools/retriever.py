from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.vectorstore import get_vectorstore
from pathlib import Path

def ingest_pdfs(data_path="data/medical_papers"):
    vectorstore = get_vectorstore()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    for pdf in Path(data_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        vectorstore.add_documents(chunks)

    vectorstore.persist()
