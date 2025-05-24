import os
from dotenv import load_dotenv
import gradio as gr
import shutil

from openai import OpenAI
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

# Cargar y etiquetar documentos
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

# División y embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=10)
splits = splitter.split_documents(pages)

embedding = OpenAIEmbeddings(openai_api_key=api_key)
persist_directory = "docs/chroma"
shutil.rmtree(persist_directory, ignore_errors=True)

vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_directory)

# Configurar modelo y retriever
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="similarity")
)

# Prompt personalizado
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Usando únicamente el siguiente contexto, responde la pregunta de forma concisa.
Si la respuesta no se encuentra en el contexto, responde: "No tengo información suficiente en los documentos proporcionados."

Contexto: {context}

Pregunta: {question}

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

# Función de respuesta
def responder(pregunta):
    try:
        output = qa_chain.invoke({"query": pregunta})
        return output["result"]
    except Exception as e:
        return f"Error: {e}"

# Interfaz Gradio
gr.Interface(
    fn=responder,
    inputs=gr.Textbox(lines=3, placeholder="Haz una pregunta sobre los documentos..."),
    outputs="text",
    title="Chatbot RAG con GPT-4o-mini",
    description="Haz preguntas basadas en los documentos PDF/TXT de la carpeta ./Datos."
).launch()
