import faiss
import sys
sys.stdout.reconfigure(encoding='utf-8')

try:
    index = faiss.read_index("email.index")
    print(f"FAISS index: {index.ntotal} vectors")
    print(f"Dimension: {index.d}")
except Exception as e:
    print(f"Error: {e}")
