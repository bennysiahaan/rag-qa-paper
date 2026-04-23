from argparse import ArgumentParser
import time

from loaders.retrieval import Retriever
from loaders.generation import Generator

def main():
    parser = ArgumentParser()
    parser.add_argument("--preload", type=str, required=True)
    parser.add_argument("--cleanup_on_init", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--prefetch_limit", type=int, default=200)
    args = parser.parse_args()
    
    retriever = Retriever(top_k=args.top_k,
                          preload_pdf_dirpath=args.preload,
                          cleanup_on_init=args.cleanup_on_init)
    generator = Generator()

    time.sleep(1.0)

    while True:
        print("\nAsk:")
        query = input("> ")
        if query.lower() in ["quit", "exit"]:
            print("Stopping.")
            break

        docs = retriever.search(query, using="hybrid", prefetch_limit=args.prefetch_limit)
        print(f"Retrieved {len(docs)} documents.")

        response = generator.query_single_with_context(query, docs)
        print(">>", response.content)

if __name__ == "__main__":
    main()