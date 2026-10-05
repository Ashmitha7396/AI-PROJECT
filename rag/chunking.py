from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Path to our agriculture document
file_path = Path("data/crops/rice.txt")

# Read the document
text = file_path.read_text(encoding="utf-8")


# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# Split the document into chunks
chunks = text_splitter.split_text(text)


# Display the results
print("=" * 60)
print("RAG CHUNKING TEST")
print("=" * 60)

print(f"\nOriginal document length: {len(text)} characters")
print(f"Number of chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print("\n" + "-" * 60)
    print(f"CHUNK {i}")
    print("-" * 60)
    print(chunk)