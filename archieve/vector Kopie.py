from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pdfplumber
import re
import os 

input_datei = "a-practical-guide-to-building-agents.pdf"
db_location = "./chroma_db"
chunk_size = 500 # Größe der Textabschnitte, die in die Datenbank eingefügt werden

add_data = not os.path.exists(db_location) # Überprüfen, ob der Ordner für die Datenbank existiert, wenn nicht, wird er erstellt
if add_data:
    # PDF-Dokument lesen und in Text umwandeln
    with pdfplumber.open(input_datei) as pdf:
        text = ""   
        for page in pdf.pages:
            text += page.extract_text()

    # Erst nach Absätzen splitten
    paragraphs = re.split(r'\n\s*\n', text) # sucht nur nach Zeilenumbruch, dann Leerzeichen, dann Zeilenumbruch. Also einem Absatz.
    chunks = []
    current_chunk = ""

    # Paragraphen werden an die Größe von chunk_size angepasst
    # Wenn Paragraph zu klein ist, werden mehrere Absätze zu einem Chunk zusammengefasst
    # Wenn Paragraph zu lang ist, wird er in kleinere Abschnitte nach Sätzen aufgeteilt
    for para in paragraphs: # Wenn Absatz zu lang, wird er in kleinere Abschnitte aufgeteilt, wenn Abschnitt zu klein ist, wird er zum aktuellen Chunk hinzugefügt
        # Wenn Absatz zu groß, nach Sätzen splitten
        if len(para) > chunk_size: # Wenn der Absatz größer als chunk_size ist, wird er in Sätze aufgeteilt
            sentences = re.split(r'(?<=[.!?]) +', para) # ?<= schaut, ob vor der aktuellen Position ein Satzzeichen steht (lookbehind). Sucht auch nach einem oder mehreren Leerzeichen nach dem Satzzeichen.
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= chunk_size: # Solange der aktuelle Chunk plus der neue Satz nicht größer als chunk_size ist, wird der Satz hinzugefügt
                    current_chunk += sent + " "
                else:
                    if current_chunk: # Wenn current_chunk nicht leer ist, wird es zu chunks hinzugefügt
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " " # Tritt ein, wenn currentchunk leer ist, aber sent zu lang.
            if current_chunk: # Für den letzten Chunk, falls er nicht leer ist
                chunks.append(current_chunk.strip())
                current_chunk = ""
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size: # Wenn der aktuelle Chunk plus der neue Absatz nicht größer als chunk_size ist, wird der Absatz hinzugefügt. +2 für die zwei Zeilenumbrüche
                current_chunk += para + "\n\n" # Fügt zeilenumbrüche hinzu, um Absätze zu trennen
            else:
                if current_chunk: # Wenn current_chunk nicht leer ist, wird es zu chunks hinzugefügt
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n" # Tritt ein, wenn currentchunk leer ist, aber para zu lang.
    if current_chunk: # Für den letzten Chunk, falls er nicht leer ist
        chunks.append(current_chunk.strip())

    documents = [Document(page_content=chunk, metadata={"chunk": idx}) for idx, chunk in enumerate(chunks)] # Chunks werden in Langchain Dokumente umgewandelt und mit einem Index versehen
    embeddings = OllamaEmbeddings(model="mxbai-embed-large") # Ollama Embeddings für die Vektorisierung der Dokumente
    db = Chroma.from_documents(documents, embeddings, persist_directory=db_location) # Datenbank wird erstellt und die Dokumente werden eingefügt. Persist_directory gibt an, wo die Datenbank dauerhaft gespeichert wird 
    print(f"Die Datenbank wurde erfolgreich unter {db_location} erstellt.")
    print("Anzahl gespeicherter Dokumente:", db._collection.count()) # Anzahl der gespeicherten Dokumente in der Datenbank ausgeben
else:
    # Nur laden, nicht neu erstellen
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=db_location, embedding_function=embeddings)
    print(f"Die Datenbank existiert bereits unter {db_location}.")
    print("Anzahl gespeicherter Dokumente:", db._collection.count())

