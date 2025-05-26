import os
from dotenv import load_dotenv
import shutil
import csv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Cargar documentos desde carpeta local
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

# Procesar los documentos
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

splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=10)
splits = splitter.split_documents(pages)

embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_directory = "docs/chroma"
shutil.rmtree(persist_directory, ignore_errors=True)

vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_directory)

# Preparar el retriever
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(
        search_type="mmr",
    )
)


# Prompt personalizado
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Responde a la siguiente pregunta exclusivamente usando el contexto proporcionado.
    Si el contexto no contiene información clara y directa para responder, debes decir:
    "No tengo información suficiente en los documentos proporcionados."

    Sé preciso. No asumas ni inventes nada que no esté explícitamente presente.

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
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# Leer preguntas y escribir resultados
input_file = "C:/Users/David/Desktop/Chatbot/test_questions.txt"
output_file = "resultados_chatbot_score.csv"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_type = "desconocido"
results = []

for line in lines:
    line = line.strip()
    if line.startswith("# type:"):
        current_type = line.split(":", 1)[1].strip()
    elif line:
        pregunta = line
        try:
            respuesta = qa_chain.invoke({"query": pregunta})["result"]
        except Exception as e:
            respuesta = f"ERROR: {e}"
        results.append({"tipo": current_type, "pregunta": pregunta, "respuesta": respuesta})

# Guardar en CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["tipo", "pregunta", "respuesta"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"✅ Resultados guardados en: {output_file}")