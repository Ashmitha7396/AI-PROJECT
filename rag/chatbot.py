from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Connect to our agriculture ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="agriculture_knowledge_v2",
    embedding_function=embeddings,
    persist_directory="database/chroma_db_v2"
)


# --------------------------------------------------
# 3. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 4. Ask the farmer
# --------------------------------------------------

question = "What type of fertilizer is suitable for crops?"


# --------------------------------------------------
# 5. Retrieve relevant agriculture information
# --------------------------------------------------

documents = retriever.invoke(question)


# --------------------------------------------------
# 6. Create context from retrieved documents
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# --------------------------------------------------
# 7. Create AI prompt
# --------------------------------------------------

prompt = f"""
You are an AI-powered agriculture assistance assistant.

Your job is to help farmers understand agriculture-related
information clearly and safely.

Use the knowledge retrieved from the agriculture database
to answer the farmer's question.

IMPORTANT RULES:

1. Use the provided agriculture knowledge as your main source.
2. Do not invent facts.
3. Do not invent pesticide prices, fertilizer prices,
   supplier names, or government benefits.
4. If the required information is not available,
   clearly say that the information is not available
   in the current knowledge base.
5. For pesticide or chemical recommendations, advise
   farmers to follow product labels and local agricultural
   recommendations.
6. Give a simple answer that a farmer can understand.

AGRICULTURE KNOWLEDGE:

{context}

FARMER QUESTION:

{question}

ANSWER:
"""


# --------------------------------------------------
# 8. Ask Ollama / Llama 3.2
# --------------------------------------------------

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 9. Display the answer
# --------------------------------------------------

print("=" * 60)
print("🌱 AI-POWERED SMART AGRICULTURE ASSISTANCE")
print("=" * 60)

print("\n👨‍🌾 Farmer's Question:")
print(question)

print("\n🤖 AI Assistant:")
print(response["message"]["content"])

print("\n📚 Sources Used:")

for document in documents:
    print("-", document.metadata.get("source"))