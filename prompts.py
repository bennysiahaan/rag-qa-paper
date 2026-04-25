import time

from loaders.retrieval import Retriever
from loaders.generation import Generator

class PromptController:
    def __init__(self,
                 model: str,
                 top_k: int,
                 mode: str,
                 num_ctx: int,
                 preload_pdf_dirpath: str,
                 cleanup_on_init: bool,
                 prefetch_limit: int,
                 markdown_path: str,
                 answer_path: str,
                 using_markdown: bool = False):
        self.top_k = top_k
        self.mode = mode if not using_markdown else "rag"
        self.prefetch_limit = prefetch_limit
        self.using_markdown = using_markdown
        self.markdown_path = markdown_path
        self.answer_path = answer_path

        self.retriever = Retriever(top_k=top_k,
                                   preload_pdf_dirpath=preload_pdf_dirpath,
                                   cleanup_on_init=cleanup_on_init)
        self.generator = Generator(model=model, num_ctx=num_ctx)

        time.sleep(1.0)

    def prompt(self, query: str):
        if query.lower() in ["quit", "exit"]:
            print("Stopping.")
            return 0, None
        
        if self.mode == "rag":
            print("Searching for documents...")
            docs = self.retriever.search(query,
                                        using="hybrid",
                                        prefetch_limit=self.prefetch_limit)
            print(f"Retrieved {len(docs)} documents.")

            print("Thinking...")
            start = time.perf_counter()
            response = self.generator.query_single_with_context(query, docs)
            end = time.perf_counter()
            print(f"Thought for {(end-start):.4f} seconds.")
        else:
            response = self.generator.query_single(query)

        print(">>", response.content)

        return 1, response.content
    
    def prompt_with_markdown(self):
        with open(self.markdown_path, "r", encoding="utf-8") as f:
            query = f.read()
            f.close()
        code, response = self.prompt(query)
        if code == 0:
            return "exit"
        if response:
            with open(self.answer_path, "w", encoding="utf-8") as f:
                f.write(str(response))
                f.close()

        return "success"

    def listen(self):
        while True:
            if self.using_markdown:
                self.prompt_with_markdown()
                break
            
            print("\nAsk:")
            query = input("> ")
            
            code, _ = self.prompt(query)
            if code == 0:
                break