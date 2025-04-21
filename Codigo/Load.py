from dotenv import load_dotenv

from openai import OpenAI
import os

# Aqui se carga un .env donde se encuentra la clave API
load_dotenv()

# Aqui obtenemos la clave guardada en el .env
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
  api_key=api_key
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)

# pip install PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

# Ruta donde están guardados los documentos
base_folder = r'C:\Users\David\Desktop\Chatbot\Datos'

# Lista para cargar los documentos (limpiamos antes de empezar)
loaders = []

# Conjunto para almacenar rutas procesadas y evitar duplicados (limpiamos antes de empezar)
processed_files = set()

# Aqui lo que hacemos es recorrer la carpeta de los documentos y los incorporamos para el Chatbot
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)

    # Comprobar si es una carpeta (solo las carpetas tienen documentos)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Verificar si ya hemos procesado este archivo (evitar duplicados)
            if file_path in processed_files:
                continue

            # Marcar este archivo como procesado
            processed_files.add(file_path)

            # Cargar PDFs si son válidos
            if file_name.lower().endswith('.pdf'):
                try:
                    print(f"📝 Cargando documento PDF: {file_path}")
                    loaders.append(PyPDFLoader(file_path))  # Añadimos el cargador de PDF
                except Exception as e:
                    print(f"⚠️ Error al leer el archivo PDF {file_path}: {e}")

            # Cargar archivos de texto (.txt)
            elif file_name.lower().endswith('.txt'):
                print(f"📝 Cargando documento TXT: {file_path}")
                loaders.append(TextLoader(file_path))  # Añadimos el cargador de TXT

# Cargar todos los documentos y añadir temática desde el nombre de la carpeta
# Ponemos una temática para ayudar al chatbot
pages = []
for loader in loaders:
    try:
        docs = loader.load()  # Cargamos el contenido de cada archivo
        for doc in docs:
            # El nombre de la carpeta tiene la tmática de los documentos
            file_path = doc.metadata.get("source", "")
            folder_name = os.path.basename(os.path.dirname(file_path))
            doc.metadata["tematica"] = folder_name  # Guardar como temática (nombre de la carpeta)
            pages.append(doc)  # Añadir documento a la lista
    except Exception as e:
        print(f"⚠️ Error al cargar el documento: {e}")


print(f"✅ Se han cargado {len(pages)} documentos (PDFs y TXT).")

# Parte para dividir el documento en partes
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Se encarga de dividir los textos en fragmentos pequeños para mejorar la búsqueda y análisis
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,  # Tamaño máximo de cada fragmento
    chunk_overlap=10,
    separators=["\n\n", "\n", r"(?<=\.)", " ", ""]  # Separadores para dividir el texto
)
splits = r_splitter.split_documents(pages)  # Aplicar la división a los documentos cargados

# pip install -U langchain-openai langchain-community
# Embedding de nuestro documento para crear representaciones vectoriales
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key="OPENAI_API_KEY")

# Vectorizamos el documento
# En caso de haber ejecutado antes, borramos lo que había en la base de datos (para evitar datos duplicados)
import shutil
persist_directory = "docs/chroma/"
shutil.rmtree(persist_directory, ignore_errors=True)

# pip install chromadb
from langchain_community.vectorstores import Chroma

# Creamos una base de datos de vectores donde guardamos los documentos
vectordb = Chroma.from_documents(
    documents=splits,  # Los documentos ya divididos en partes
    embedding=embedding,  # Usamos el embedding para convertir el texto a vectores
    persist_directory=persist_directory  # Directorio donde se guardará la base de datos persistente
)
print(vectordb._collection.count())  # Mostramos cuántos documentos están indexados en la base de datos

# Prueba del código

# Respuestas
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key
)

# Función para imprimir los documentos de forma ordenada y con su temática
def pretty_print_docs(docs):
    for i, doc in enumerate(docs):
        tema = doc.metadata.get("tematica", "Sin temática")  # Obtener la temática del documento que hemos puesto antes
        print(f"\n--- Documento {i+1} (Temática: {tema}) ---")
        print(doc.page_content[:500])  # Muestra solo los primeros 500 caracteres

# Comprimir los documentos para mejorar la eficiencia al hacer preguntas
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)  # Usamos un extractor de cadenas con el modelo de lenguaje
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="similarity")  # Usamos búsqueda por similitud en la base de datos - el que mejor resultados me ha dado
)

# Hacer una consulta sobre los documentos
pregunta = "donde estan las aulas de clase y en que numero de bloque"
compressed_docs = compression_retriever.invoke(pregunta)  # Obtenemos los documentos relevantes
pretty_print_docs(compressed_docs)  # Mostramos los documentos encontrados

# Responder la pregunta usando el modelo de lenguaje y los documentos comprimidos
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define un prompt simple para QA (preguntas y respuestas)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Usando únicamente el siguiente contexto, responde la pregunta de forma concisa.
Si la respuesta no se encuentra en el contexto, responde: "No tengo información suficiente en los documentos proporcionados."

Contexto: {context}

Pregunta: {question}

Respuesta:
"""
)

# Crea una cadena de QA usando el modelo y los documentos comprimidos
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 'stuff' es para respuestas directas, 'map_reduce' es para casos con muchos documentos
    retriever=compression_retriever,
    return_source_documents=True,  # Retorna también los documentos utilizados para la respuesta
    chain_type_kwargs={"prompt": prompt}  # Pasamos el prompt de la pregunta
)

# Hacer la pregunta y obtener la respuesta
output = qa_chain.invoke({"query": pregunta})
respuesta = output["result"]

print("Respuesta:", respuesta)  # Imprimimos la respuesta obtenida

# Mostrar las fuentes utilizadas para obtener la respuesta
fuentes = output["source_documents"]
for doc in fuentes:
    print("\n📄 Fuente:")
    print(doc.page_content)  # Mostramos el contenido de los documentos fuente utilizados para la respuesta
print(doc.page_content)