from argparse import ArgumentParser
from pathlib import Path

from loaders.retrieval import Retriever
from loaders.generation import Generator
from prompts import PromptController

def main():
    parser = ArgumentParser()
    parser.add_argument("--preload", type=str, default=None)
    parser.add_argument("--cleanup_on_init", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--prefetch_limit", type=int, default=200)
    parser.add_argument("--markdown", action="store_true", default=False)
    args = parser.parse_args()

    markdown_path = Path(__file__).parent / "prompt.md"

    controller = PromptController(top_k=args.top_k,
                                  preload_pdf_dirpath=args.preload,
                                  cleanup_on_init=args.cleanup_on_init,
                                  prefetch_limit=args.prefetch_limit,
                                  markdown_path=str(markdown_path),
                                  using_markdown=args.markdown)

    controller.listen()

if __name__ == "__main__":
    main()