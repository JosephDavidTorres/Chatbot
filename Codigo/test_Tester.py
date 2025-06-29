import os
import csv
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Cargar API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Rutas
base_folder = "C:/Users/David/Desktop/Chatbot/Datos"
persist_directory = "docs/chroma/"

# Cargar documentos
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

# Dividir documentos
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
splits = splitter.split_documents(pages)
shutil.rmtree(persist_directory, ignore_errors=True)

# Embeddings y vectorstore
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)

batch_size = 100
for i in range(0, len(splits), batch_size):
    batch = splits[i:i + batch_size]
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]
    vectordb.add_texts(texts=texts, metadatas=metadatas)

# LLMs
llm_clasificador = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
llm_respuestas = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# Prompt clasificación
clasificacion_prompt = PromptTemplate(
    input_variables=["pregunta"],
    template="""
Eres un asistente que clasifica preguntas realizadas por estudiantes dentro del ámbito universitario.

Clasifica la siguiente pregunta en una de estas tres categorías, siguiendo estas instrucciones:

- CLARA: si la pregunta se entiende por sí sola y puede responderse directamente en el contexto de una universidad.
- AMBIGUA: si la pregunta no está clara por sí sola, tiene múltiples posibles significados, depende de una conversación anterior.
- FUERA DE CONTEXTO: si no está relacionada con la universidad.

Pregunta: "{pregunta}"

Devuelve solo una de estas palabras en mayúsculas: CLARA, AMBIGUA o FUERA DE CONTEXTO.
"""
)
clasificacion_chain = clasificacion_prompt | llm_clasificador

# Prompt QA
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Usando únicamente el siguiente contexto, responde la pregunta de forma concisa.
Si la respuesta no se encuentra en el contexto, responde: "No tengo información suficiente en los documentos proporcionados."
Si recibes 2 preguntas, la primera es solo para darte el contexto de que ha preguntado antes, no hace falta responder la primera pregunta

Contexto: {context}

Pregunta: {question}

Respuesta:
"""
)

# Ejecutar tests
test_folder = "Test"
output_folder = "Resultados"
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(test_folder):
    if not file_name.endswith(".txt"):
        continue

    test_path = os.path.join(test_folder, file_name)
    test_name = os.path.splitext(file_name)[0]

    with open(test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for search_type in ["mmr", "similarity"]:
        retriever = vectordb.as_retriever(search_type=search_type, search_kwargs={"k": 6})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_respuestas,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_prompt}
        )

        results = []
        memoria = {"ultima_pregunta": None, "ultima_respuesta": None}
        current_type = "respuesta_directa"

        for line in lines:
            line = line.strip()
            if line.startswith("# type:"):
                current_type = line.split(":", 1)[1].strip()
                continue
            elif not line:
                continue

            pregunta_original = line
            categoria = clasificacion_chain.invoke(pregunta_original).content.strip().upper()

            if categoria == "CLARA":
                output = qa_chain.invoke({"query": pregunta_original})
                respuesta = output["result"]
                fuente = output["source_documents"][0].metadata.get("source", "-") if output["source_documents"] else "-"
                if "No tengo información suficiente" not in respuesta:
                    memoria["ultima_pregunta"] = pregunta_original
                    memoria["ultima_respuesta"] = respuesta
                else:
                    memoria["ultima_pregunta"] = None
                    memoria["ultima_respuesta"] = None

            elif categoria == "AMBIGUA":
                if memoria["ultima_pregunta"] and memoria["ultima_respuesta"]:
                    pregunta_con_contexto = (
                        f"Pregunta actual: {pregunta_original}\n"
                        f"Pregunta anterior: {memoria['ultima_pregunta']}"
                    )
                    output = qa_chain.invoke({"query": pregunta_con_contexto})
                    respuesta = output["result"]
                    fuente = output["source_documents"][0].metadata.get("source", "-") if output["source_documents"] else "-"
                    if "No tengo información suficiente" in respuesta:
                        memoria["ultima_pregunta"] = None
                        memoria["ultima_respuesta"] = None
                else:
                    respuesta = "Tu pregunta es ambigua. Por favor, intenta especificarla mejor o haz una pregunta más clara."
                    fuente = "-"
                    memoria["ultima_pregunta"] = None
                    memoria["ultima_respuesta"] = None

            elif categoria == "FUERA DE CONTEXTO":
                respuesta = "Esta pregunta no está relacionada con los documentos universitarios proporcionados."
                fuente = "-"
                memoria["ultima_pregunta"] = None
                memoria["ultima_respuesta"] = None

            results.append({
                "tipo": current_type,
                "pregunta": pregunta_original,
                "respuesta": respuesta,
                "fuente": fuente
            })

        output_file = os.path.join(output_folder, f"resultados_{test_name}_{search_type}.csv")
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["tipo", "pregunta", "respuesta", "fuente"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print(f"✅ Resultados guardados en: {output_file}")
