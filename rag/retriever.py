from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load the same embedding model used when creating the database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to our new agriculture ChromaDB
vector_store = Chroma(
    collection_name="agriculture_knowledge_v2",
    embedding_function=embeddings,
    persist_directory="database/chroma_db_v2"
)


# Create the retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Test question
question = "What type of fertilizer is suitable for crops?"


# Retrieve relevant information
documents = retriever.invoke(question)


print("=" * 60)
print("RAG RETRIEVER TEST")
print("=" * 60)

print(f"\nQuestion: {question}")
print(f"\nRelevant documents found: {len(documents)}")


for i, document in enumerate(documents, start=1):

    print("\n" + "-" * 60)
    print(f"RESULT {i}")
    print("-" * 60)

    print(document.page_content)

    print("\nSource:")
    print(document.metadata.get("source"))