from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=groq_model,
    model_kwargs={
        "prompts": """
You are Ungacode bot, a helpful and slightly witty assistant trained on a space station 14 forked codebase called Civ14. 
Speak informally, like a programmer explaining things to another programmer. Use humor where appropriate, but never be sarcastic or rude.
You can also attempt to match the tone of the user interacting with you.
""",
    },
)
embedding = HuggingFaceEmbeddings(model_name=embed_model)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="chunk")
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


@app.get("/", response_class=HTMLResponse)
async def ask_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, query: str = Form(...)):
    response = qa_chain.invoke({"query": query})
    return templates.TemplateResponse(
        "index.html", {"request": request, "query": query, "response": response}
    )


@app.get("/learn", response_class=HTMLResponse)
async def learn_page(request: Request):
    return templates.TemplateResponse("learn.html", {"request": request})


@app.post("/learn", response_class=HTMLResponse)
async def learn_fact(request: Request, fact: str = Form(...)):
    vector = embedding.embed_documents([fact])[0]
    index.upsert(
        vectors=[(str(uuid.uuid4()), vector)],
        namespace="learned",
        metadata={"chunk": fact},
    )
    return templates.TemplateResponse("learn.html", {"request": request, "saved": fact})


@app.get("/memories", response_class=HTMLResponse)
async def memory_page(request: Request):
    stats = index.describe_index_stats(namespace="learned")
    return templates.TemplateResponse(
        "memories.html", {"request": request, "stats": stats}
    )
