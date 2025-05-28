import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.qa_engine import answer_question

if __name__=="__main__":
    while True:
        query = input("Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        print("\n"+ answer_question(query)+ "\n")