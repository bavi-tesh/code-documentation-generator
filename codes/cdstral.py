from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

file = open("content.txt", "r", encoding = 'utf-8')
template = """Question: {question}
Answer: Generate unique test cases in swagger. Include all possible forms of inputs, give a descriptive output"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="codestral")

chain = prompt | model


result = chain.invoke({file.read()})
print(result)

