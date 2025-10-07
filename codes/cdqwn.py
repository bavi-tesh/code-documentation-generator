from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

file = open("Python_code_data.txt", "r", encoding = 'utf-8')
template = """Question: {question}
Answer: Generate documentation for each code include complexities from the file"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="codeqwen")

chain = prompt | model


result = chain.invoke({file.read()})
print(result)

