import os
from dotenv import load_dotenv
import shutil
import csv
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Cargar API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Cargar documentos
base_folder = "C:/Users/David/Desktop/Chatbot/Datos"
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
                except:
                    pass
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
    except:
        pass

splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
splits = splitter.split_documents(pages)

embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_directory = "docs/chroma"
shutil.rmtree(persist_directory, ignore_errors=True)

vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_directory)

# Preparar retriever y modelo
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="similarity")
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Responde a la siguiente pregunta exclusivamente usando el contexto proporcionado.
Si el contexto incluye una conversación anterior, puedes usarla solo si es claramente relevante para entender la nueva pregunta.

Si no hay suficiente información para responder de forma precisa, di:
"No tengo información suficiente en los documentos proporcionados."

Sé conciso, preciso y no inventes nada que no esté en el contexto.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Validación de longitud de pregunta
def es_pregunta_valida(texto, memoria_activa):
    palabras = re.findall(r'\b\w+\b', texto)
    if memoria_activa:
        return True  # si hay memoria previa, dejamos pasar aunque sea corta
    return len(palabras) >= 4

# Memoria de conversación anterior
memoria = {"ultima_pregunta": None, "ultima_respuesta": None}

# Leer preguntas
input_file = "C:/Users/David/Desktop/Chatbot/test_questions.txt"
output_file = "resultados_chatbot_mejorado_mmr.csv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_type = "desconocido"
results = []

for line in lines:
    line = line.strip()
    if line.startswith("# type:"):
        current_type = line.split(":", 1)[1].strip()
    elif line:
        pregunta_original = line

        # 🧠 Verificar si hay memoria activa
        memoria_activa = memoria["ultima_pregunta"] and memoria["ultima_respuesta"]

        # ✅ Validar longitud de pregunta (solo si no hay memoria)
        if not es_pregunta_valida(pregunta_original, memoria_activa):
            respuesta = "Por favor, especifica mejor tu pregunta."
            fuente = "-"

        else:

            # Añadir memoria si existe
            if memoria["ultima_pregunta"] and memoria["ultima_respuesta"]:
                pregunta = f"En la conversación anterior se dijo:\nUsuario: {memoria['ultima_pregunta']}\nChatbot: {memoria['ultima_respuesta']}\n\nAhora el usuario pregunta:\n{pregunta_original}"
            else:
                pregunta = pregunta_original

            documentos = retriever.invoke(pregunta)
            contexto = "\n\n".join([doc.page_content for doc in documentos])

            if not contexto.strip() or len(contexto.split()) < 50:
                respuesta = "No tengo información suficiente en los documentos proporcionados."
                fuente = "-"
            else:
                output = qa_chain.invoke({"query": pregunta})
                respuesta = output["result"]
                fuente = output["source_documents"][0].metadata.get("source", "Documento desconocido") if output["source_documents"] else "-"

            # Actualizar memoria
            if respuesta.strip() == "No tengo información suficiente en los documentos proporcionados.":
                memoria["ultima_pregunta"] = None
                memoria["ultima_respuesta"] = None
            else:
                memoria["ultima_pregunta"] = pregunta_original
                memoria["ultima_respuesta"] = respuesta
        results.append({
            "tipo": current_type,
            "pregunta": pregunta_original,
            "respuesta": respuesta,
            "fuente": fuente
        })

# Guardar resultados
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["tipo", "pregunta", "respuesta", "fuente"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"✅ Resultados guardados en: {output_file}")
