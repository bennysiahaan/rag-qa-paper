from argparse import ArgumentParser
from pathlib import Path

import ollama
import questionary

from prompts import PromptController

DEFAULT_MODEL_GENERATION = "gemma4:e4b"

def select_model() -> str:
    client = ollama.Client()
    models = [m.model for m in client.list().models if m.model is not None]
    
    print(f"Found {len(models)} available models.")
    if not models:
        raise RuntimeError("No model available.")
    
    return questionary.select(
        "Select a model:",
        choices=models,
        use_shortcuts=True,
    ).ask()

def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--preload", type=str, default=None)
    parser.add_argument("--cleanup_on_init", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--prefetch_limit", type=int, default=200)
    parser.add_argument("--markdown", type=str, nargs="?", const="<default>", default=None)
    parser.add_argument("--num_ctx", type=int, default=None)
    args = parser.parse_args()

    markdown_path = args.markdown
    using_markdown = True if markdown_path is not None else False
    if markdown_path is None or markdown_path == "<default>":
        markdown_path = Path(__file__).parent / "prompt.md"

    answer_path = Path(__file__).parent / "answer.md"

    model = select_model()

    controller = PromptController(top_k=args.top_k,
                                  model=model,
                                  mode=args.mode,
                                  num_ctx=args.num_ctx,
                                  preload_pdf_dirpath=args.preload,
                                  cleanup_on_init=args.cleanup_on_init,
                                  prefetch_limit=args.prefetch_limit,
                                  markdown_path=str(markdown_path),
                                  answer_path=str(answer_path),
                                  using_markdown=using_markdown)

    controller.listen()

if __name__ == "__main__":
    main()