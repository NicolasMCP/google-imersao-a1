'''
Nota: 
$ uv init google-imersao-a1
$ cd google-imersao-a1/
$ uv venv
$ source .venv/bin/activate
$ code .
$ uv pip install langchain-google-genai google-generativeai
$ uv pip install ipykernel
$ uv pip install jupyter
$ uv pip install python-dotenv
'''

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Acessa a chave da variável de ambiente
gemini_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0,          # Criatividade das respostas (0 a 1) 0 é mais preciso
    api_key=gemini_key
)

# resposta  = llm.invoke("Como usar RAG (de IA)? O que eu precisso saber?")
# print(resposta.content)

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)


class TriagemOutput(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOutput = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

if __name__ == "__main__":
    llm_triagem = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature = 0,          # Criatividade das respostas (0 a 1) 0 é mais preciso
        api_key=gemini_key
    )
    
    triagem_chain = llm_triagem.with_structured_output(TriagemOutput)

    testes = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como Faço?",
        "Posso reembolsar curosos ou treinamentos da Alura?",
        "Quantas capivaras tem no rio pinheiros?"
    ] 
    for msg in testes:
        resultado = triagem(msg)
        print(f"Mensagem: {msg}\nResultado: {resultado}\n") 
