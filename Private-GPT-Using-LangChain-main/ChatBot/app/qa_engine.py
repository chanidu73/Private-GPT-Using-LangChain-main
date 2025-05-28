import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.zephyr_loader import load_zephyr
from models.embedder import load_embedder
from app.retriever import Retriever

embedder= load_embedder()
retriever = Retriever(embedder)
generator = load_zephyr()

def answer_question(question):
    docs = retriever.retrieve(question)
    context = '\n\n'.join(docs)

    prompt = f"""Use the following context to answer the question:
    
    Context:
    {context}

    Question:
    {question}

    Answer:

    """
    result = generator(prompt , max_new_tokens=256 , do_sample =True)
    return result[0]["generated_text"]