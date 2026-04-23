from argparse import ArgumentParser
import time

from langchain_ollama import ChatOllama

from loaders.retrieval import Retriever

def main():
    generation_model = "qwen3.5:9b"

    parser = ArgumentParser()
    parser.add_argument("--preload", type=str, required=True)
    args = parser.parse_args()
    
    retriever = Retriever(preload_pdf_dirpath=args.preload,
                          cleanup_on_init=True)
    time.sleep(1.0)

    while True:
        print("\nSearch:")
        query = input("> ")

        results = retriever.search(query, using="hybrid")
        for i, result in enumerate(results):
            print(f"-------------- Document {i+1} --------------\n\n")
            print(f"File: {result.source}\n\nContent:\n{result.context}\n\n")

if __name__ == "__main__":
    main()