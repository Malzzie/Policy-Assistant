from pathlib import Path

from src.loaders import load_documents
from src.chunking import chunk_documents


BASE_DIR = Path(__file__).resolve().parents[1]
POLICIES_DIR = BASE_DIR / "data" / "policies"

print("BASE_DIR:", BASE_DIR)
print("POLICIES_DIR:", POLICIES_DIR)
print("Exists?:", POLICIES_DIR.exists())

docs = load_documents(POLICIES_DIR)
print(f"\nLoaded {len(docs)} source docs/pages")

chunks = chunk_documents(
    docs=docs,
    chunk_size=700,
    chunk_overlap=100,
)

print(f"Created {len(chunks)} chunks\n")

# Show the first few chunks so you can inspect the structure
for chunk in chunks[:10]:
    print("chunk_id:", chunk["chunk_id"])
    print("doc_id:", chunk["doc_id"])
    print("title:", chunk["title"])
    print("source:", chunk["source"])
    print("page:", chunk["page"])
    print("section:", chunk["section"])
    print("text preview:", chunk["text"][:200].replace("\n", " "))
    print("-" * 80)