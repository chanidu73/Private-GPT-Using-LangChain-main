import faiss 
import os 
import pickle 
from sentence_transformers.util import cos_sim

class Retriever:
    def __init__(self , embedder , index_path = "db/faiss_store/index.faiss"):
        self.embedder = embedder
        self.index_path = index_path
        self.load_index()


    def load_index(self):
        if os.path.exists(self.index_path):
            self.index= faiss.read_index(self.index_path)
            with open("db/faiss_store/id_map.pkl" , "rb")as f:
                self.id_map = pickle.load(f)
        else:
            self.index =None
            self.id_map =[]

    def retrieve(self , query , top_k=3):
        if self.index is None:
            return []
        query_vec = self.embedder.encode([query])
        scores , indices = self.index.search(query_vec , top_k)

        return [self.id_map[i] for i in  indices[0]]
    