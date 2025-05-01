import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
import uuid
import json
from colorama import Fore, Back, Style

# logging.basicConfig(level=logging.DEBUG)
# --- Setup ---
load_dotenv()
print(
    Fore.LIGHTWHITE_EX,
    "Loading...",
    Fore.RESET,
)
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
        print(Fore.RED, "Index not found, run updater first!", Fore.RESET)
        return
    index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=model, text_key="chunk")
    retriever = vector_store.as_retriever()
    client = Groq(api_key=groq_api_key)

    while True:
        print(
            Fore.GREEN,
            "Vibebot is ready! Type 'exit' to quit, start your query with 'learn:' to insert knowledge.",
            Fore.RESET,
        )
        query = input(Fore.GREEN + "Input: " + Fore.RESET)
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
        if query.lower().startswith("learn:"):
            fact = query[len("learn:") :].strip()
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
            metadata_file = "metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = []
            existing_metadata.append(metadata_entry)
            with open(metadata_file, "w") as f:
                json.dump(existing_metadata, f)
            print("Learned new knowledge: ", Fore.CYAN, f"{fact}", Fore.RESET)
        else:
            embedding = HuggingFaceEmbeddings(model_name=embed_model)
            response_text = response.choices[0].message.content
            qa_text = f"Q: {query}\\nA: {response_text}"
            vector = embedding.embed_documents([qa_text])[0]
            index.upsert(
                vectors=[(str(uuid.uuid4()), vector)],
                namespace="qa_history",
                metadata={"chunk": qa_text},
            )
            print(Fore.GREEN, "Response:", Fore.YELLOW, response_text, Fore.RESET)


if __name__ == "__main__":
    main()
