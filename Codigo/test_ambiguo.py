import os
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# Cargar API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Rutas
base_folder = "C:/Users/David/Desktop/Chatbot/Datos"
persist_directory = "docs/chroma/"
input_file = "test_questions.txt"
output_file = "resultados_clasificacion.csv"

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

# Dividir documentos en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=1000)
splits = splitter.split_documents(pages)

# Crear embeddings y base de datos vectorial
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)

# Modelo LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)

# Prompt para clasificar preguntas
clasificacion_prompt = PromptTemplate(
    input_variables=["pregunta"],
    template="""
Clasifica la siguiente pregunta en una de estas categorías:

- CLARA: si puede responderse de forma precisa por sí sola, sin depender de contexto previo.
- AMBIGUA: si necesita más información, depende de una conversación anterior o es demasiado general.
- FUERA DE CONTEXTO: si no tiene relación con el ámbito universitario o los documentos proporcionados.

Pregunta: "{pregunta}"

Devuelve solo una de estas palabras en mayúsculas: CLARA, AMBIGUA o FUERA DE CONTEXTO.
"""
)
clasificacion_chain = LLMChain(llm=llm, prompt=clasificacion_prompt)

# Prompt para responder
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Usando únicamente el siguiente contexto, responde la pregunta de forma concisa.
Si la respuesta no se encuentra en el contexto, responde: "No tengo información suficiente en los documentos proporcionados."

Contexto: {context}

Pregunta: {question}

Respuesta:
"""
)

# Configuración del retriever
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# Cadena de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

# Evaluación de preguntas
results = []
memoria = {"ultima_pregunta": None, "ultima_respuesta": None}

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_type = "respuesta_directa"
for line in lines:
    line = line.strip()
    if line.startswith("# type:"):
        current_type = line.split(":", 1)[1].strip()
    elif line:
        pregunta_original = line

        # Clasificación
        categoria = clasificacion_chain.run(pregunta_original).strip().upper()

        if categoria == "AMBIGUA":
            memoria_activa = memoria["ultima_pregunta"] and memoria["ultima_respuesta"]

            if memoria_activa:
                # Construimos nueva pregunta combinando contexto anterior
                pregunta_con_contexto = f"{memoria['ultima_pregunta']} -> {pregunta_original}"
                output = qa_chain.invoke({"query": pregunta_con_contexto})
                respuesta = output["result"]
                fuente = output["source_documents"][0].metadata.get("source", "-") if output[
                    "source_documents"] else "-"

                memoria["ultima_pregunta"] = pregunta_original
                memoria["ultima_respuesta"] = respuesta
            else:
                respuesta = "Tu pregunta es ambigua. Por favor, intenta especificarla mejor o haz una pregunta más clara."
                fuente = "-"
                memoria["ultima_pregunta"] = None
                memoria["ultima_respuesta"] = None

            results.append({
                "tipo": current_type,
                "pregunta": pregunta_original,
                "respuesta": respuesta,
                "fuente": fuente
            })
            continue

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
            continue

        # Si es clara, responder normalmente
        output = qa_chain.invoke({"query": pregunta_original})
        respuesta = output["result"]
        fuente = output["source_documents"][0].metadata.get("source", "-") if output["source_documents"] else "-"

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
