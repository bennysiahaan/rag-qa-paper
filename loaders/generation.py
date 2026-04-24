from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

class Generator:
    def __init__(self,
                 temperature: float = 0.0):
        # self.model = "qwen3.5:9b"
        # self.model = "qwen2.5:3b"
        # self.model = "qwen2.5:7b"
        self.model = "llama3.1:8b"
        self.system_template = (
            "You are a helpful assistant who answers truthfully based on"
            " the retrieved documents. You have to say that you do not know if"
            " you cannot find the factual and proven answer from the documents."
        )
        self.system_template_with_context = self.system_template + "\n\nDocuments:\n{context}"
        self.human_template = "{text}"
        self.chat_model = ChatOllama(model=self.model,
                                     temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("human", self.human_template),
        ])
        self.chain = self.prompt | self.chat_model
        
        self.prompt_with_context = ChatPromptTemplate.from_messages([
            ("system", self.system_template_with_context),
            ("human", self.human_template),
        ])
        self.chain_with_context = self.prompt_with_context | self.chat_model
    
    def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{i+1} | source={d.metadata.get('source', '?')}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )
    
    def query_single(self, query: str):
        return self.chain.invoke({"text": query})
    
    def query_single_with_context(self, query: str, docs: list[Document]):
        return self.chain_with_context.invoke({
            "context": self._format_docs(docs),
            "text": query,
        })