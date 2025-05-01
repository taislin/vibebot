import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
# --- Setup ---
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def main():
    model = HuggingFaceEmbeddings(
        model_name=embed_model, model_kwargs={"device": "cpu"}
    )
    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name not in pc.list_indexes().names():
        print("Index not found, run updater first!")
        return
    index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=model, text_key="chunk")
    retriever = vector_store.as_retriever()
    client = Groq(api_key=groq_api_key)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=groq_model, client=client)

    while True:
        query = input("Ask about the codebase (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break
        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        # Call Groq API directly
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
        print("Response:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
