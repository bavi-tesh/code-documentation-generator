from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

file = open("opnstck.txt", "r", encoding = 'utf-8')
template = """Question: {question}
Answer: Generate docstrings after each line of code and give the  enterprise grade documentationfor the codes"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="codegemma")

chain = prompt | model


result = chain.invoke({file.read()})
print(result)

