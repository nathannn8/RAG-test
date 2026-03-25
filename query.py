import ollama
import chromadb

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="documents")

question = input("Ask a question: ")

response = ollama.embeddings(model="nomic-embed-text", prompt= question)
embedding = response["embedding"]

results = collection.query(
    query_embeddings=[embedding],
    n_results=1
)

for doc in results["documents"][0]:
    print(doc)
    print("---")