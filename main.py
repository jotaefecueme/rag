import os
import json
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

# ---------------------- CONFIGURATION ----------------------
def load_config():
    """Loads configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

config = load_config()

VECTORSTORE_FOLDER = config.get("VECTORSTORE_FOLDER")
EMBEDDING_MODEL_NAME = config.get("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = config.get("LLM_MODEL_NAME")
RETRIEVAL_TOP_K = config.get("RETRIEVAL_TOP_K", 5)
TEMPERATURE = config.get("TEMPERATURE", 0)
KEEP_ALIVE = config.get("KEEP_ALIVE", 3600)

PROMPT_TEMPLATE = """Eres un modelo de RAG para responder a preguntas de clientes realizadas por teléfono.
Utiliza la documentación proporcionada para dar una respuesta concisa a la pregunta.
Recuerda que el usuario no tiene acceso a la documentación, así que no hagas mención de páginas ni apartados de la documentación.
Si no encuentras información relevante, responde con [NO_INFO].
No eres un asistente, no tienes que hacer *small talk* ni mantener una conversación. Tu única función es devolver siempre la información obtenida de la documentación que pueda ayudar a la consulta.
La documentación disponible es la siguiente: {context}

Pregunta: {question}
"""



# ---------------------- VECTORSTORE LOADING ----------------------
def load_vectorstore():
    """Loads the FAISS vectorstore with embeddings."""
    try:
        if not os.path.exists(VECTORSTORE_FOLDER):
            raise FileNotFoundError(f"Vectorstore folder not found: {VECTORSTORE_FOLDER}")
        print(f"Loading vectorstore from: {VECTORSTORE_FOLDER}")
        embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return FAISS.load_local(VECTORSTORE_FOLDER, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

# ---------------------- RAG PIPELINE INITIALIZATION ----------------------
def initialize_rag_chain(retriever):
    """Initializes the RAG pipeline using an LLM and a retriever."""
    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=TEMPERATURE, keep_alive=KEEP_ALIVE)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ---------------------- FUNCTION TO PROCESS USER QUESTIONS ----------------------
def process_user_question(user_question, retriever):
    """Process the user's question and return the response."""
    try:
        rag_chain = initialize_rag_chain(retriever)
        return rag_chain.invoke(user_question)
    except Exception as e:
        print(f"Error processing question: {e}")
        return "Error: Unable to process the question."

# ---------------------- FASTAPI SETUP ----------------------
app = FastAPI()

# Pydantic model to handle incoming requests
class QuestionRequest(BaseModel):
    question: str

# Initialize vectorstore at the start
vectorstore = load_vectorstore()

if not vectorstore:
    raise Exception("Could not initialize vectorstore. Exiting.")

retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG question answering service!"}

@app.post("/ask/")
def ask_question(request: QuestionRequest):
    """Endpoint to process user questions."""
    try:
        response = process_user_question(request.question, retriever)
        return {"question": request.question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the question: {e}")

# ---------------------- EXECUTION ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
