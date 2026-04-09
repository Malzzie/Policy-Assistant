from pathlib import Path
from src.loaders import load_documents

BASE_DIR = Path(__file__).resolve().parents[1]
POLICIES_DIR = BASE_DIR / "data" / "policies"

print("BASE_DIR:", BASE_DIR)
print("POLICIES_DIR:", POLICIES_DIR)
print("Exists?:", POLICIES_DIR.exists())

docs = load_documents(POLICIES_DIR)

print(f"\nLoaded {len(docs)} docs\n")

for doc in docs:
    print(doc["source"], "|", doc["title"], "|", doc["page"])