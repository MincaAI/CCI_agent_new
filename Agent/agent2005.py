import os
import sys
import time
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from Agent.storage import store_lead_to_google_sheet  #
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# === 1. Charger les variables d'environnement ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialisation ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
llm = ChatOpenAI(temperature=0.5, model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = PineconeVectorStore(index=index, embedding=embeddings).as_retriever()
long_term_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="memory")

# Mémoire conversationnelle par utilisateur (chat history)
chat_histories = {}
inactivity_event = threading.Event()

def load_evenements_context():
    try:
        with open("Data/evenements_structures.txt", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""


def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    
def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()
    
evenements_context = load_evenements_context()


def get_full_conversation(session_id: str) -> str:
    docs = long_term_store.similarity_search(" ", k=100)
    user_docs = [doc for doc in docs if doc.metadata.get("user_id") == session_id]
    user_docs.sort(key=lambda d: d.metadata.get("timestamp", ""))
    return "\n".join(doc.page_content for doc in user_docs)


def save_message_to_long_term_memory(role: str, message: str, session_id: str):
    now = datetime.now(timezone.utc).isoformat()
    doc = Document(
        page_content=f"{role} : {message}",
        metadata={"user_id": session_id, "timestamp": now, "type": role}
    )
    long_term_store.add_documents([doc])


def has_calendly_link(text: str) -> bool:
    return "https://calendly.com/" in text


def extract_lead_info(history: str) -> dict:
    template = load_extraction_prompt_template()
    prompt = template.replace("{{history}}", history)

    try:
        response = llm.invoke(prompt).content
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("\n", 1)[0]
        response = response.strip()

        import json
        data = json.loads(response)
        data["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return data
    except Exception as e:
        return {
            "prenom": "inconnu",
            "nom": "inconnu",
            "entreprise": "inconnu",
            "email": "inconnu",
            "interet": "inconnu",
            "score": 1,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }



def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


chain = RunnableWithMessageHistory(llm, get_session_history)
prompt_template = load_prompt_template()

def agent_response(user_input: str, session_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    history = get_full_conversation(session_id)

    # ➕ Ajouter la recherche de contexte CCI à partir de la question posée
    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"

    # ➕ Injecter dans le prompt
    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{history}}", history or "[Aucune conversation précédente]")\
                            .replace("{{cci_context}}", base_cci_context)\
                            .replace("{{evenements_context}}", evenements_context or "[Aucun événement à afficher]")
    
    reply = chain.invoke(input=prompt, config={"configurable": {"session_id": session_id}})
    reply_text = reply.content if hasattr(reply, "content") else str(reply)

    save_message_to_long_term_memory("Utilisateur", user_input, session_id)
    save_message_to_long_term_memory("Agent", reply_text, session_id)
    return reply_text


def surveillance_inactivite(session_id: str, timeout=50):
    while True:
        inactivity_event.clear()
        if not inactivity_event.wait(timeout):
            try:
                history = get_full_conversation(session_id)
                if has_calendly_link(history):
                    lead = extract_lead_info(history)
                    if lead.get("prenom") != "inconnu" and lead.get("email") != "inconnu":
                        store_lead_to_google_sheet(lead)
            except Exception:
                pass
            sys.exit()
        time.sleep(1)


if __name__ == "__main__":
    session_id = "session_003"
    threading.Thread(target=surveillance_inactivite, args=(session_id,), daemon=True).start()

    try:
        while True:
            user_input = input("💬 Vous : ")
            inactivity_event.set()
            reply = agent_response(user_input, session_id=session_id)
            print(f"\n🧠 Agent :\n{reply}\n")
    except KeyboardInterrupt:
        pass

