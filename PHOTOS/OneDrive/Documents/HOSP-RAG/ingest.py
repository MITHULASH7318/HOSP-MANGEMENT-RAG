"""
ingest.py — Rebuild the FAISS knowledge base from documents.
Run from the project root: python ingest.py [--force]
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_engine import initialize_rag, get_document_stats, DOCUMENTS_DIR


def main():
    force = "--force" in sys.argv

    print("=" * 60)
    print("  🏥 Hospital RAG — Knowledge Base Ingestion")
    print("=" * 60)

    # ✅ Pass documents_dir explicitly (fixes the missing-argument bug)
    stats = get_document_stats(documents_dir=DOCUMENTS_DIR)

    print(f"\n📁 Documents dir : {Path(DOCUMENTS_DIR).resolve()}")
    print(f"📄 Found {stats['total_files']} file(s):")
    for fname in stats["filenames"]:
        print(f"   • {fname}")

    if stats["total_files"] == 0:
        print("\n❌ No documents found! Add .txt or .pdf files to:")
        print(f"   {Path(DOCUMENTS_DIR).resolve()}")
        sys.exit(1)

    print(f"\n🔨 Building FAISS index (force={force})...")
    try:
        initialize_rag(force_rebuild=force or True)
        print("\n✅ Knowledge base ready!")
        print("🚀 Run:  streamlit run app.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()