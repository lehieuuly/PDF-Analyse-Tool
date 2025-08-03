from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os 

input_dir = "./pdf_dir"
db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=db_location, embedding_function=embeddings)

loader = PyPDFDirectoryLoader(input_dir)
documents_load = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " "]
)

# Alle existierenden chunk_ids aus der Datenbank holen
existing_chunk_ids = set()
results = db._collection.get(include=["metadatas"])
for metadata in results["metadatas"]:
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        existing_chunk_ids.add(chunk_id)

# Neue Chunks sammeln
add_document = []
existing = 0
for doc in documents_load:
    chunks = text_splitter.split_text(doc.page_content)
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc.metadata['source']}:{doc.metadata['page']}:{doc.metadata['page_label']}:{idx}"
        if chunk_id not in existing_chunk_ids:
            add_document.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata['source'],
                    "page": doc.metadata['page'],
                    "page_label": doc.metadata['page_label'],
                    "chunk": idx,
                    "chunk_id": chunk_id
                }
            ))
        else:
            existing += 1

# Hier solltest du nur add_documents verwenden, nicht from_documents!
if add_document:
    db.add_documents(add_document)

print(f"Bereits existierende Chunks: {existing}")
print(f"Neue Chunks zum Hinzufügen: {len(add_document)}")
print(f"Anzahl der Chunks in der Datenbank: {db._collection.count()}")