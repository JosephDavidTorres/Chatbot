import os
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

#Cargar la API (En caso de que no exista se necesitará conseguir una)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Comandos para asegurarse que el script funcione en cualquier dispositivo
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

current_file_path = os.path.abspath(__file__)

root_folder = os.path.basename(os.path.dirname(os.path.dirname(current_file_path)))

base_folder = os.path.join(desktop_path, root_folder, "Datos")

# Para ver si no va
if not os.path.exists(base_folder):
    raise FileNotFoundError(f"La carpeta '{base_folder}' no existe.")

persist_directory = "docs/chroma/"

#Cargar los documentos
loaders = []
processed_files = set()
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path in processed_files:
                continue
            processed_files.add(file_path)
            if file_name.lower().endswith(".pdf"):
                try:
                    loaders.append(PyPDFLoader(file_path))
                except Exception:
                    continue
            elif file_name.lower().endswith(".txt"):
                loaders.append(TextLoader(file_path))

pages = []
for loader in loaders:
    try:
        docs = loader.load()
        for doc in docs:
            file_path = doc.metadata.get("source", "")
            folder_name = os.path.basename(os.path.dirname(file_path))
            doc.metadata["tematica"] = folder_name
            pages.append(doc)
    except Exception:
        continue

# Generar los embeddings y la fragmentacion
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
splits = splitter.split_documents(pages)

shutil.rmtree(persist_directory, ignore_errors=True)
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

batch_size = 100
for i in range(0, len(splits), batch_size):
    batch = splits[i:i + batch_size]
    vectordb.add_texts([doc.page_content for doc in batch], [doc.metadata for doc in batch])

# Los 2 modelos, uno de clasificación y otro para las respuestas
llm_clasificador = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
llm_respuestas = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# Prompt de Clasificacion
clasificacion_prompt = PromptTemplate(
    input_variables=["pregunta"],
    template="""
Eres un asistente que clasifica preguntas realizadas por estudiantes dentro del ámbito universitario.

Clasifica la siguiente pregunta en una de estas tres categorías, siguiendo estas instrucciones:
- CLARA: si la pregunta se entiende por sí sola y puede responderse directamente en el contexto de una universidad. Esto incluye preguntas sobre lugares ("¿Dónde está la biblioteca?"), fechas académicas, procedimientos comunes, asignaturas, matrículas o documentación.
- AMBIGUA: si la pregunta no está clara por sí sola, tiene múltiples posibles significados, depende de una conversación anterior, o se refiere a algo sin contexto suficiente (por ejemplo: "¿Es obligatorio?", "¿Dónde es?", "¿Cuándo se hace?"). Si se refiere a zonas que puede haber en una universidad no se considera ambigua sino Clara
- FUERA DE CONTEXTO: si no está relacionada con la universidad, estudios, clases o procedimientos académicos (por ejemplo: "¿Cuántos habitantes tiene Francia?", "¿Cuál es el precio del Bitcoin?").

Pregunta: "{pregunta}"

Devuelve solo una de estas palabras en mayúsculas: CLARA, AMBIGUA o FUERA DE CONTEXTO.
"""
)
clasificacion_chain = clasificacion_prompt | llm_clasificador

#Prompt para las respuestas
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Usando únicamente el siguiente contexto, responde la pregunta de forma concisa.
Si la respuesta no se encuentra en el contexto, responde: "No tengo información suficiente en los documentos proporcionados."
Responde solo a la pregunta actual. Usa la pregunta anterior solo como ayuda para interpretarla si es ambigua y que te ayude a responderla.

Contexto: {context}

Pregunta: {question}

Respuesta:
"""
)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_respuestas,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# FastAPI
app = FastAPI()
frontend_path = pathlib.Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Permitir CORS (para conexión con frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memoria para el chatbot
memoria = {"ultima_pregunta": None, "ultima_respuesta": None}

class PreguntaRequest(BaseModel):
    pregunta: str

#Lógica del Cahtbot
@app.post("/preguntar")
def preguntar(req: PreguntaRequest):
    pregunta = req.pregunta.strip()
    categoria = clasificacion_chain.invoke(pregunta).content.strip().upper()
    if categoria == "CLARA":
        output = qa_chain.invoke({"query": pregunta})
        respuesta = output["result"]
        if "No tengo información suficiente" not in respuesta:
            memoria["ultima_pregunta"] = pregunta
            memoria["ultima_respuesta"] = respuesta
        else:
            memoria["ultima_pregunta"] = None
            memoria["ultima_respuesta"] = None

    elif categoria == "AMBIGUA" and memoria["ultima_pregunta"] and memoria["ultima_respuesta"]:
        pregunta_con_contexto = (
            f"Pregunta actual: {pregunta}\n"
            f"Pregunta anterior: {memoria['ultima_pregunta']}"
        )
        output = qa_chain.invoke({"query": pregunta_con_contexto})
        respuesta = output["result"]
        if "No tengo información suficiente" in respuesta:
            memoria["ultima_pregunta"] = None
            memoria["ultima_respuesta"] = None
    elif categoria == "AMBIGUA":
        respuesta = "Tu pregunta es ambigua. Por favor, intenta especificarla mejor."
        memoria["ultima_pregunta"] = None
        memoria["ultima_respuesta"] = None
    else:
        respuesta = "Esta pregunta no está relacionada con los documentos universitarios proporcionados."
        memoria["ultima_pregunta"] = None
        memoria["ultima_respuesta"] = None

    return {"respuesta": respuesta, "categoria": categoria}

import uvicorn

@app.get("/")
def serve_index():
    return FileResponse(frontend_path / "index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
