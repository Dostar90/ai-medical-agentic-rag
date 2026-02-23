from langchain_community.vectorstores import Chroma


from rag.embeddings import get_embedding_model

def get_vectorstore(persist_dir="chroma_db"):
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
