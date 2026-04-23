import time

from langchain_ollama import ChatOllama

from loaders.retrieval import Retriever

def main():
    retrieval_model = "bge-m3:latest"
    generation_model = "qwen3.5:9b"

    preload_pdf_dirpath = "./dev/pdf"
    
    retriever = Retriever(model=retrieval_model,
                          preload_pdf_dirpath=preload_pdf_dirpath)
    time.sleep(1.0)

    while True:
        print("\nSearch:")
        query = input("> ")

        results = retriever.search(query)
        for i, result in enumerate(results):
            print(f"-------------- Document {i+1} --------------\n\n")
            print(f"File: {result.source}\n\nContent:\n{result.context}\n\n")

if __name__ == "__main__":
    main()