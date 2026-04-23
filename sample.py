import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model_name = "gpt-4o-mini"
chat_model = ChatOpenAI(model=model_name, api_key=api_key)

### Prompt Template
template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{user_input}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), ("human", human_template)
])

messages = chat_prompt.format_messages(input_language="English",
                                       output_language="Japanese",
                                       user_input="Hello, world!")
response = chat_model.invoke(messages)
print(response.content)

### Output Parser
class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split("answer =")
template = (
    "You are a helpful assistant that solves math problems and shows your work."
    " Output each step then return the answer in the following format: answer = <answer here>."
    " Make sure to output answer in all lowercases and to have exactly one space and one equal"
    " sign following it."
)
human_template = "{problem}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), ("human", human_template)
])

messages = chat_prompt.format_messages(problem="2x^2 - 5x + 3 = 0")
result = chat_model.invoke(messages)
parsed = AnswerOutputParser().parse(result.content)
steps, answer = parsed

print("Steps:", steps)
print("Answer:", answer)

### Chaining
class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")

template = (
    "You are a helpful assistant who generates comma separated lists."
    " A user will pass in a category, and you should generate 5 objects in that category"
    " in a comma separated list."
    " ONLY return a comma separated list, and nothing more."
)
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template), ("human", human_template)
])

chain = chat_prompt | chat_model | CommaSeparatedListOutputParser()
result = chain.invoke({"text": "colors"})
print(result)