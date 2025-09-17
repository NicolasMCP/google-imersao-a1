'''
Agentes.py
Autor: Nícolas Ramos
Data: 2025-09-14

Versão: SEM LIMITE - Uso do Ollama localmente

Nota: 
$ uv init google-imersao-a1
$ cd google-imersao-a1/
$ uv venv
$ source .venv/bin/activate
###$ uv pip install langchain-google-genai google-generativeai
$ uv pip install langchain-ollama
$ uv pip install ipykernel
$ uv pip install jupyter
###$ uv pip install python-dotenv
$ code .
'''

import os
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()

# O Ollama não requer uma chave de API como o Gemini
# Acessa a chave da variável de ambiente
# gemini_key = os.getenv('GEMINI_API_KEY')

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature = 0,          # Criatividade das respostas (0 a 1) 0 é mais preciso
#     api_key=gemini_key
# )

TRIAGEM_PROMPT = """Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento.
Dada a mensagem do usuário, retorne SOMENTE um JSON com:
{{
  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",
  "urgencia": "BAIXA" | "MEDIA" | "ALTA",
  "campos_faltantes": ["..."]
}}

Regras:
- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").
- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").
- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").

Analise a mensagem e decida a ação mais apropriada."""

class TriagemOutput(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)  

# def triagem(mensagem: str) -> Dict:
#     saida: TriagemOutput = triagem_chain.invoke([
#         SystemMessage(content=TRIAGEM_PROMPT),
#         HumanMessage(content=mensagem)
#     ])
def triagem(mensagem: str, llm: ChatOllama) -> Dict:
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", TRIAGEM_PROMPT),
        ("human", "{mensagem}")
    ])
    # Constrói a cadeia (chain) que liga o prompt, o LLM e o parser de saída estruturada
    triagem_chain = prompt_template | llm.with_structured_output(TriagemOutput)
    # Invoca a cadeia com a mensagem do usuário
    saida: TriagemOutput = triagem_chain.invoke({"mensagem": mensagem})
    # Retorna o resultado como um dicionário
    return saida.model_dump()


if __name__ == "__main__":
    llm_triagem = ChatOllama(
        model="gemma3:4b",   
        temperature=0        # Garante respostas precisas
    )

    testes = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como Faço?",
        "Posso reembolsar curosos ou treinamentos da Alura?",
        "Quantas capivaras tem no rio pinheiros?"
    ]

    for msg in testes:
        try:
            resultado = triagem(msg, llm_triagem)
            print(f"Mensagem: {msg}\nResultado: {resultado}\n") 
        except Exception as e:
            print(f"Erro ao processar a mensagem: {msg}\nDetalhe do Erro: {e}\n")
