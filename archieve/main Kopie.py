from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory=db_location, embedding_function=embeddings)
retriever = db.as_retriever() # Retriever erstellen

model = OllamaLLM(model="llama3.2")

with open("prompt.txt", "r", encoding="utf-8") as f:
    template = f.read()

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # Zuerst wird Prompt erstellt, dann wird es mit dem Modell verbunden. Kurzform für das Hintereinanderschalten von Verarbeitungsschritten

# Interaktive Benutzereingabe
print("Willkommen zum PDF Analyse Tool!")
while True:
    user_input = input("Geben Sie Ihre Frage ein (oder 'q' zum Beenden): ")
    if user_input.lower() == 'q':
        break
    relevant_docs = retriever.invoke(user_input) # Relevante Chunks abrufen
    pdf_document = "\n".join([doc.page_content for doc in relevant_docs])
    result = chain.invoke({"pdf_document": pdf_document, "user_question": user_input}) # Chain mit den gefundenen Chunks und der Frage aufrufen
    print(result) # Ausgabe des Ergebnisses der Kette für die Benutzereingabe