from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

file = open("content.txt", "r", encoding = 'utf-8')
template = """Question: {question}
Answer: Generate unique test cases for testing purposes for the codes give a descriptive output"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="codellama")

chain = prompt | model


result = chain.invoke({file.read()})
print(result)

