import os
import sys
import time
import threading
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from Agent.storage import store_lead_to_google_sheet
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

class StreamPrintCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# === 1. Charger les variables d'environnement ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialisation ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
llm = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamPrintCallback()]
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = PineconeVectorStore(
    index=index,
    embedding=embeddings
).as_retriever(search_kwargs={"k": 2})
long_term_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="memory")

# Mémoire conversationnelle par utilisateur avec ConversationSummaryBufferMemory
chat_memories = {}
inactivity_event = threading.Event()

def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    
def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()

def get_conversation_memory(chat_id: str) -> ConversationSummaryBufferMemory:
    """Récupère ou crée une mémoire ConversationSummaryBuffer pour un chat_id donné"""
    if chat_id not in chat_memories:
        # Créer une nouvelle mémoire avec limite de tokens (working memory seulement)
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,  # Ajustez selon vos besoins
            return_messages=True,
            memory_key="chat_history"
        )
        
        chat_memories[chat_id] = memory
    
    return chat_memories[chat_id]

def save_message_to_long_term_memory(role: str, message: str, chat_id: str):
    """Sauvegarde les messages dans Pinecone pour persistance"""
    now = datetime.now(timezone.utc).isoformat()
    doc = Document(
        page_content=f"{role} : {message}",
        metadata={"chat_id": chat_id, "timestamp": now, "type": role}
    )
    long_term_store.add_documents([doc])

def has_calendly_link(text: str) -> bool:
    return "https://calendly.com/" in text

def extract_lead_info(memory: ConversationSummaryBufferMemory) -> dict:
    """Extrait les informations de lead à partir de la mémoire"""
    template = load_extraction_prompt_template()
    
    # Récupérer l'historique formaté depuis la mémoire
    history_text = memory.buffer
    
    prompt = template.replace("{{history}}", history_text)

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

prompt_template = load_prompt_template()

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    
    # Récupérer la mémoire conversationnelle pour ce chat (working memory seulement)
    memory = get_conversation_memory(chat_id)
    
    # Récupérer l'historique de conversation depuis la mémoire working
    conversation_history = memory.buffer if isinstance(memory.buffer, str) else str(memory.buffer)
    
    # Récupérer le contexte CCI à partir de la question posée
    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"

    # Construire le prompt avec le contexte working memory + CCI
    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{history}}", conversation_history or "[Aucune conversation précédente]")\
                            .replace("{{cci_context}}", base_cci_context)
    print("\n🧾 CONTENU DE conversation_history :\n")
    print(conversation_history)
    
    print("\n🧠 Agent :\n", end="", flush=True)
    
    # Générer la réponse
    reply = await llm.ainvoke(prompt)
    reply_text = reply.content if hasattr(reply, "content") else str(reply)

    # Ajouter les messages à la mémoire (elle gèrera automatiquement la summarisation)
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(reply_text)
    
    # Debug: afficher l'état de la mémoire
    print(f"\n🔍 [DEBUG] Mémoire buffer length: {len(memory.buffer)} caractères")
    print(f"🔍 [DEBUG] Nombre de messages en mémoire: {len(memory.chat_memory.messages)}")
    
    # Sauvegarder aussi dans Pinecone pour persistance (optionnel pour debug)
    save_message_to_long_term_memory("Utilisateur", user_input, chat_id)
    save_message_to_long_term_memory("Agent", reply_text, chat_id)
    
    return reply_text

def surveillance_inactivite(chat_id: str, timeout=50):
    while True:
        inactivity_event.clear()
        if not inactivity_event.wait(timeout):
            try:
                # Utiliser la mémoire working au lieu de l'historique complet
                memory = get_conversation_memory(chat_id)
                if has_calendly_link(memory.buffer):
                    lead = extract_lead_info(memory)
                    store_lead_to_google_sheet(lead)
            except Exception:
                pass
            sys.exit()
        time.sleep(1)

# Fonction pour tester et afficher l'état de la mémoire
def debug_memory_state(chat_id: str):
    """Fonction utile pour débugger l'état de la mémoire"""
    if chat_id in chat_memories:
        memory = chat_memories[chat_id]
        print(f"\n=== DEBUG MÉMOIRE POUR {chat_id} ===")
        print(f"Buffer content:\n{memory.buffer}")
        print(f"Nombre de messages: {len(memory.chat_memory.messages)}")
        print(f"Longueur du buffer: {len(memory.buffer)} caractères")
        print("="*50)
    else:
        print(f"Aucune mémoire trouvée pour {chat_id}")
        
if __name__ == "__main__":
    import uuid
    chat_id = str(uuid.uuid4())  # 👈 Generate a unique ID per session
    print(f"🆔 Session Chat ID: {chat_id}")
    threading.Thread(target=surveillance_inactivite, args=(chat_id,), daemon=True).start()

    try:
        while True:
            user_input = input("💬 Vous : ")
            inactivity_event.set()
            reply = asyncio.run(agent_response(user_input, chat_id=chat_id))
            print(f"\n🧠 Agent :\n{reply}\n")
    except KeyboardInterrupt:
        pass