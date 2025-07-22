import os
import argparse
import shutil
from pathlib import Path
from itertools import islice


from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = Path("data")
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 80
BATCH_SIZE = 5000

def batched(iterable, size=BATCH_SIZE):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

def main() -> None:
    args = parse_args()
    if args.reset:
        clear_database()
    docs = load_documents()
    # No chunking
    add_to_chroma(docs)

def parse_args():
    parser = argparse.ArgumentParser(description="CSV -> Chroma ingestor")
    parser.add_argument(
        "--reset", action="store_true", help="Delete the existing Chroma DB first"
    )
    return parser.parse_args()

# LOAD
def load_documents() -> list[Document]:
    """
    Walk DATA_PATH, load each *.csv independently
    so different schemas are preserved
    """
    documents: list[Document] = []
    for csv_path in DATA_PATH.glob("*.csv"):
        loader = CSVLoader(str(csv_path), csv_args={"delimiter": ","})
        file_docs = loader.load()
        # Attach filename so filtering can be done
        for i, doc in enumerate(file_docs):
            doc.metadata["source_file"] = csv_path.name
            doc.metadata["row"] = i
        documents.extend(file_docs)
    print(f"Loaded {len(documents)} rows from {len(list(DATA_PATH.glob('*.csv')))} CSVs")
    return documents

# NO CHUNKING - Keep POI rows intact
def split_documents(documents: list[Document]) -> list[Document]:
    """
    No chunking - keep each CSV row intact as a complete POI entry
    """
    print(f"Keeping {len(documents)} POI documents intact (no chunking)")
    return documents

# STORE / UPDATE
def add_to_chroma(documents: list[Document]) -> None:
    """
    Add documents to Chroma
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    
    # Attach IDs
    documents = attach_document_ids(documents)
    
    existing_ids = set(db.get(include=[])["ids"])
    print(f"DB currently contains {len(existing_ids)} documents")
    
    new_documents = [doc for doc in documents if doc.metadata["id"] not in existing_ids]
    if not new_documents:
        print("No new documents to add")
        return
    
    print(f"Adding {len(new_documents)} new documents")
    for start in range(0, len(new_documents), BATCH_SIZE):
        batch = new_documents[start : start + BATCH_SIZE]
        ids = [doc.metadata["id"] for doc in batch]
        db.add_documents(batch, ids=ids)
        print(f" inserted rows {start}-{start+len(batch)-1}")
    
    print("DB updated & persisted")

def attach_document_ids(documents: list[Document]) -> list[Document]:
    """
    Attach unique IDs to documents (simplified since no chunking)
    """
    for doc in documents:
        file = doc.metadata["source_file"]
        row = doc.metadata["row"]
        doc.metadata["id"] = f"{file}::{row}"
    return documents

def clear_database() -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Existing Chroma DB removed")

if __name__ == "__main__":
    main()