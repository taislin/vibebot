from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
texts = ["Sample code text" * 100] * 100
start_time = time.time()
embeddings = embed_model.get_text_embedding_batch(texts)
print(f"Time taken: {time.time() - start_time} seconds")
