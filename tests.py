# test_code.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()  # Cargar las variables de entorno


# Función para cargar el VectorStore de documentos
def load_vectorstore():
    persist_directory = "docs/chroma/"
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb.as_retriever()


# Función para hacer las consultas al modelo
def get_answer(query, retriever):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    result = qa.run(query)
    return result


# Test: Llamada para obtener respuestas a preguntas
def run_tests():
    # Cargar el VectorStore una sola vez
    retriever = load_vectorstore()

    # Lista de preguntas para probar
    preguntas = [
        "¿Dónde se encuentra el aula de matemáticas?",
        "¿Cuál es la legislación vigente sobre derechos laborales?",
        "¿Qué dice la constitución acerca de la educación?"
    ]

    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")
        respuesta = get_answer(pregunta, retriever)
        print("Respuesta:", respuesta)


# Ejecutar las pruebas
if __name__ == "__main__":
    run_tests()
