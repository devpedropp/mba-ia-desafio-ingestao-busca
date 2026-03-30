from search import search_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def main():    
    while True:
        user_input = input("Digite sua pergunta (ou 'sair' para encerrar o chat): ")
        if user_input.lower() == "sair":
            break

        chain = search_prompt(user_input)
    
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0.3)
        promptChain = ChatPromptTemplate.from_template(chain) | llm | StrOutputParser()
        response = promptChain.invoke({})
        print(response)


if __name__ == "__main__":
    main()