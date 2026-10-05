from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Find all agriculture documents
# --------------------------------------------------

data_folder = Path("data")
files = list(data_folder.rglob("*.txt"))

print("=" * 60)
print("AGRICULTURE KNOWLEDGE INGESTION")
print("=" * 60)

print(f"\nDocuments found: {len(files)}")


# --------------------------------------------------
# 2. Read all documents
# --------------------------------------------------

all_texts = []
all_metadatas = []

for file_path in files:

    text = file_path.read_text(encoding="utf-8")

    all_texts.append(text)

    all_metadatas.append({
        "source": str(file_path)
    })

    print(f"✓ Loaded: {file_path}")


# --------------------------------------------------
# 3. Split all documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []
all_chunk_metadata = []

for text, metadata in zip(all_texts, all_metadatas):

    chunks = text_splitter.split_text(text)

    for chunk in chunks:
        all_chunks.append(chunk)
        all_chunk_metadata.append(metadata)


print(f"\nTotal chunks created: {len(all_chunks)}")


# --------------------------------------------------
# 4. Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 5. Create a fresh ChromaDB collection
# --------------------------------------------------

print("\nCreating ChromaDB...")

vector_store = Chroma(
    collection_name="agriculture_knowledge_v2",
    embedding_function=embeddings,
    persist_directory="database/chroma_db_v2"
)


# --------------------------------------------------
# 6. Store all chunks
# --------------------------------------------------

print("Storing agriculture knowledge...")

vector_store.add_texts(
    texts=all_chunks,
    metadatas=all_chunk_metadata
)


# --------------------------------------------------
# 7. Finished
# --------------------------------------------------

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)

print(f"Documents loaded : {len(files)}")
print(f"Chunks stored    : {len(all_chunks)}")
print("Database         : ChromaDB")
print("Status           : READY")