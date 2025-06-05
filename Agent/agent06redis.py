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
from storage import store_lead_to_google_sheet  #
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

import redis
import pickle

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_USE_SSL", "false").lower() == "true"
)

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
summary_store = PineconeVectorStore(index=index, embedding=embeddings,namespace="summaries")

# Mémoire conversationnelle par utilisateur (chat history)
SUMMARY_INTERVAL = 30
message_counters = {}


def save_summary_to_pinecone(chat_id: str, summary: str):
    timestamp = datetime.now(timezone.utc)
    doc = Document(
        page_content=summary,
        metadata={
            "chat_id": chat_id,
            "summary_id": f"summary_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": timestamp.isoformat()
        }
    )
    summary_store.add_documents([doc])

def load_summary_from_pinecone(chat_id: str) -> str:
    docs = summary_store.similarity_search("", k=1, filter={"chat_id": chat_id})
    return docs[0].page_content if docs else ""

def generate_summary_from_memory(memory: ConversationSummaryBufferMemory, chat_id: str, max_messages: int = 10) -> str:
    recent_messages = memory.chat_memory.messages[-max_messages:]
    convo_text = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in recent_messages])
    if not convo_text.strip():
        return ""
    prompt = f"""
    Voici une conversation entre un utilisateur et un agent de la CCI France Mexique. Résume-la en identifiant :

    - Qui est l'utilisateur ? (situation, rôle, objectifs)
    - Quels besoins ou intentions a-t-il exprimés ?
    - Quels services ou ressources lui ont été proposés ?
    - Y a-t-il des éléments à suivre ou relancer ?

    Résumé en 7 phrases max. Conversation :
    \n\n{convo_text}
    """
    summary = llm.invoke(prompt).content.strip()
    save_summary_to_pinecone(chat_id, summary)
    return summary

def save_summary_memory_to_redis(chat_id, memory):
    state = {
        "messages": memory.chat_memory.messages,
        "summary": memory.moving_summary_buffer
    }
    redis_client.set(chat_id, pickle.dumps(state))

def load_summary_memory_from_redis(chat_id):
    data = redis_client.get(chat_id)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
        memory_key="chat_history"
    )
    if data:
        state = pickle.loads(data)
        memory.chat_memory.messages = state.get("messages", [])
        memory.moving_summary_buffer = state.get("summary", "")
    return memory


def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    
def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()
    

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
        
prompt_template = load_prompt_template()

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    memory = load_summary_memory_from_redis(chat_id)
    short_term_memory = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in memory.buffer])
    long_term_memory = load_summary_from_pinecone(chat_id)
    

    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"
    

    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{short_term_memory}}", short_term_memory or "[Aucune mémoire courte]")\
                            .replace("{{long_term_memory}}", long_term_memory or "[Aucune mémoire longue]")\
                            .replace("{{cci_context}}", base_cci_context)
    
    print("\n🧠 Mémoire courte terme :\n", short_term_memory or "[Aucune mémoire courte]")
    print("\n📚 Mémoire longue terme :\n", long_term_memory or "[Aucune mémoire longue]")
    print("\n📋 Contexte CCI :\n", base_cci_context, "\n")

    reply = await llm.ainvoke(prompt)
    reply_text = reply.content if hasattr(reply, "content") else str(reply)

    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(reply_text)
    save_summary_memory_to_redis(chat_id, memory)

    # Incrément du compteur et résumé auto tous les 30 messages
    message_counters[chat_id] = message_counters.get(chat_id, 0) + 2
    if message_counters[chat_id] >= SUMMARY_INTERVAL:
        message_counters[chat_id] = 0
        generate_summary_from_memory(memory,chat_id)

    return reply_text


if __name__ == "__main__":
    import uuid
    chat_id = str(uuid.uuid4())  # 👈 Generate a unique ID per session
    print(f"🆔 Session Chat ID: {chat_id}")

    try:
        while True:
            user_input = input("💬 Vous : ")
            reply = asyncio.run(agent_response(user_input, chat_id=chat_id))
            print(f"\n🧠 Agent :\n{reply}\n")
    except KeyboardInterrupt:
        pass
