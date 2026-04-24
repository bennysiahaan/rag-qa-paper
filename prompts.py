import time

from loaders.retrieval import Retriever
from loaders.generation import Generator

class PromptController:
    def __init__(self,
                 top_k: int,
                 preload_pdf_dirpath: str,
                 cleanup_on_init: bool,
                 prefetch_limit: int,
                 markdown_path: str,
                 using_markdown: bool = False):
        self.top_k = top_k
        self.prefetch_limit = prefetch_limit
        self.markdown_path = markdown_path
        self.using_markdown = using_markdown

        self.retriever = Retriever(top_k=top_k,
                                   preload_pdf_dirpath=preload_pdf_dirpath,
                                   cleanup_on_init=cleanup_on_init)
        self.generator = Generator()

        time.sleep(1.0)

    def prompt(self, query: str):
        if query.lower() in ["quit", "exit"]:
            print("Stopping.")
            return
        
        docs = self.retriever.search(query,
                                     using="hybrid",
                                     prefetch_limit=self.prefetch_limit)
        print(f"Retrieved {len(docs)} documents.")

        response = self.generator.query_single_with_context(query, docs)
        print(">>", response.content)
    
    def prompt_with_markdown(self):
        with open(self.markdown_path, "r") as f:
            query = f.read()
        self.prompt(query)

        return "success"

    def listen(self):
        while True:
            if self.using_markdown:
                return self.prompt_with_markdown()
            
            print("\nAsk:")
            query = input("> ")
            
            self.prompt(query)