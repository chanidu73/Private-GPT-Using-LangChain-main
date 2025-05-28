import os 
from tqdm import tqdm
from PyPDF2 import PdfReader
# from models.embedder import load_embedder
import faiss
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.embedder import load_embedder

# /workspaces/Private-GPT-Using-LangChain/ChatBot/models
def load_documents(folder):
    texts =[]
    for file in os.listdir(folder):
        path =os.path.join(folder , file)
        if file.endswith(".pdf"):
            reader = PdfReader(path)
            texts.append("\n".join(page.extract_text() for page in reader.pages) )
        elif file.endswith(".txt"):
            with open(path , "r" , encoding="utf-8") as f:
                texts.append(f.read())

    return texts
    
def ingest(folder):
    embedder = load_embedder()
    documents = load_documents(folder)
    embedings = embedder.encode(documents)

    index = faiss.IndexFlatL2(embedings.shape[1])
    index.add(embedings)

    os.makedirs("db/faiss_store" , exist_ok=True)
    faiss.write_index(index , "db/faiss_store/index.faiss")
    with open("db/faiss_store/id_map.pkl" , "wb") as f:
        pickle.dump(documents ,f)

if __name__ == "__main__":
    ingest("data/sample_docs")