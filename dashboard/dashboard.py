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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import json
from groq import Groq

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

model = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"})
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Load metadata for BM25
with open("./../metadata.json", "r") as f:
    raw_docs = json.load(f)

bm25_docs = [
    Document(page_content=doc["chunk"], metadata={"source": doc["file"]})
    for doc in raw_docs
]
bm25_retriever = BM25Retriever.from_documents(bm25_docs)
bm25_retriever.k = 2

embedding = HuggingFaceEmbeddings(model_name=embed_model)
dense_retrievers = []
for ns in ["learned", "code", "qa_history", ""]:
    vs = PineconeVectorStore(
        index=index, embedding=embedding, text_key="chunk", namespace=ns
    )
    dense_retrievers.append(vs.as_retriever(search_kwargs={"k": 2}))

# Combine dense + sparse retrievers
retriever = EnsembleRetriever(
    retrievers=dense_retrievers + [bm25_retriever],
    weights=[1.0] * (len(dense_retrievers) + 1),
)
client = Groq(api_key=groq_api_key)


@app.get("/", response_class=HTMLResponse)
async def ask_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, query: str = Form(...)):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content[:500] for doc in docs if doc.page_content])[
        :2000
    ]
    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role": "system",
                "content": """
You are Ungacode bot, a helpful and slightly witty assistant trained on a space station 14 forked codebase called Civ14. 
Speak informally, like a programmer explaining things to another programmer. Use humor where appropriate, but never be sarcastic or rude.
You can also attempt to match the tone of the user interacting with you.
""",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
        ],
    )
    embedding = HuggingFaceEmbeddings(model_name=embed_model)
    response_text = response.choices[0].message.content
    qa_text = f"Q: {query}\\nA: {response_text}"
    vector = embedding.embed_documents([qa_text])[0]
    qa_id = str(uuid.uuid4())
    index.upsert(
        vectors=[
            (
                qa_id,
                vector,
                {"chunk": qa_text[:500], "file": f"qa_history/{qa_id}.txt"},
            )
        ],
        namespace="qa_history",
    )
    return templates.TemplateResponse(
        "index.html", {"request": request, "query": query, "response": response_text}
    )


@app.get("/learn", response_class=HTMLResponse)
async def learn_page(request: Request):
    return templates.TemplateResponse("learn.html", {"request": request})


@app.post("/learn", response_class=HTMLResponse)
async def learn_fact(request: Request, fact: str = Form(...)):
    docs = retriever.invoke(fact)
    context = "\n".join([doc.page_content[:500] for doc in docs if doc.page_content])[
        :2000
    ]
    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {
                "role": "system",
                "content": """
You are Ungacode bot, a helpful and slightly witty assistant trained on a space station 14 forked codebase called Civ14. 
Speak informally, like a programmer explaining things to another programmer. Use humor where appropriate, but never be sarcastic or rude.
You can also attempt to match the tone of the user interacting with you.
""",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {fact}"},
        ],
    )
    model = HuggingFaceEmbeddings(
        model_name=embed_model, model_kwargs={"device": "cpu"}
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)
    knowledge_id = str(uuid.uuid4())
    embedding = model.embed_documents([fact])[0]
    metadata = {
        "chunk": fact[:500],
        "file": f"learned_knowledge/{knowledge_id}.txt",
    }
    index.upsert(
        vectors=[(knowledge_id, embedding, metadata)],
        namespace="learned",
    )
    metadata_entry = {
        "file": metadata["file"],
        "chunk": metadata["chunk"],
        "full_chunk": fact,
    }
    metadata_file = "./../metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []
    existing_metadata.append(metadata_entry)
    with open(metadata_file, "w") as f:
        json.dump(existing_metadata, f)
    return templates.TemplateResponse("learn.html", {"request": request, "saved": fact})


@app.get("/memories", response_class=HTMLResponse)
async def memory_page(request: Request):
    stats = index.describe_index_stats(namespace="learned")
    return templates.TemplateResponse(
        "memories.html", {"request": request, "stats": stats}
    )
