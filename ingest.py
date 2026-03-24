import PyPDF2
import ollama
import chromadb

def load_pdf(path):
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest(pdf_path):
    text = load_pdf(pdf_path)
    chunks = split_text(text)
    
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection(name="documents")
    
 
    for i, chunk in enumerate(chunks):
        response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        embedding = response["embedding"]
        
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    print(f"Stored {len(chunks)} chunks")

if __name__ == "__main__":
    ingest("Bean_and_Leaf_Business_Proposal.pdf")