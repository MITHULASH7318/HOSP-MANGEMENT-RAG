"""Run once to build the knowledge base: python ingest.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.rag_engine import initialize_rag, get_document_stats

stats = get_document_stats()
print(f"Found {stats['total_files']} documents: {stats['filenames']}")
if stats['total_files'] == 0:
    print("Add .txt or .pdf files to data/documents/ first")
    sys.exit(1)
print("Building index...")
initialize_rag(force_rebuild=True)
print("Done! Now run: streamlit run app.py")