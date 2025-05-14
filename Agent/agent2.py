import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from datetime import datetime
from langchain.schema import Document


# === 1. Charger les variables d’environnement ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialiser le modèle ===
llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

# === 3. Initialiser Pinecone retriever ===
retriever = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=OpenAIEmbeddings()
).as_retriever()


long_term_store = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=OpenAIEmbeddings(),
    namespace="memory"
)

# === 4. Charger les événements depuis un fichier local ===
def load_evenements_context():
    try:
        with open("evenements_structures.txt", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Aucun événement n'a pu être chargé."

evenements_context = load_evenements_context()

# === 5. Rechercher dans la base de connaissances ===
def get_cci_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# === 6. Enregistrer un message dans la mémoire long terme ===
def save_to_long_term_memory(text: str, user_id: str):
    now = datetime.utcnow().isoformat()
    doc = Document(
        page_content=text,
        metadata={
            "user_id": user_id,
            "timestamp": now,
            "type": "user_message"
        }
    )
    long_term_store.add_documents([doc])

# === 7. Récupérer la mémoire longue d’un utilisateur ===
def retrieve_long_term_memory(query: str, user_id: str) -> str:
    docs = long_term_store.similarity_search(query, k=10)
    filtered_docs = [doc for doc in docs if doc.metadata.get("user_id") == user_id]
    return "\n\n".join(doc.page_content for doc in filtered_docs) if filtered_docs else ""

# === 8. Mémoire de conversation courte (session) ===
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)


# === 6. Fonction principale de l’agent ===
def agent_response(user_input: str, user_id: str) -> str:
    base_cci_context = get_cci_context(user_input)
    long_term_context = retrieve_long_term_memory(user_input, user_id)

    prompt = f"""
Tu es un assistant intelligent et multilingue de la Chambre de Commerce et d’Industrie Franco-mexicaine.
SI un utilisateur te parle en français, réponds en français. Si c'est en espagnol, réponds en espagnol.
Mission

Voici la mémoire utilisateur long terme :
{long_term_context or '[Aucune mémoire pertinente pour cet utilisateur.]'}

Voici des informations de la base CCI :
{base_cci_context or '[Pas d’information pertinente dans la base.]'}
Répondre de manière claire, professionnelle et utile à toutes les questions portant sur :

* les services proposés (accompagnement, formations, événements, networking, soutien aux entreprises, etc.)
* les conditions et avantages d’adhésion
* les offres réservées aux membres
* les partenariats, actualités et événements à venir

Bonnes pratiques

* proposer des liens directs vers les pages utiles lorsque c’est pertinent
* indiquer où télécharger les documents ou formulaires nécessaires
* si une information n’est pas disponible, le préciser avec courtoisie et orienter l’utilisateur vers un contact de la CCI

Langue

* détecter automatiquement si la question est posée en français ou en espagnol
* répondre intégralement dans la langue détectée

Style

* ton professionnel, bienveillant et informatif
* langage clair et structuré, avec des listes à puces si besoin
* ne jamais donner d'information incertaine
* ne jamais sortir du périmètre de la CCI
  (si la question ne concerne pas la CCI, expliquer poliment que ton rôle est uniquement d’informer sur la CCI et ses services)



L'utilisateur a dit : "{user_input}"

Réponds en français, de façon professionnelle, fluide et utile. Si l'utilisateur pose une question simple (bonjour, merci, etc.), réponds naturellement sans chercher d'information.
"""

    full_prompt = f"{prompt}\nUtilisateur : {user_input}"
    reply = conversation.predict(input=full_prompt)

    # Enregistrement du message utilisateur dans la mémoire longue
    save_to_long_term_memory(f"Utilisateur : {user_input}\nRéponse de l'agent : {reply}", user_id=user_id)

    return reply

# === 10. Boucle de chat ===
if __name__ == "__main__":
    print("🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !\nTape 'exit' pour quitter.\n")
    
    # This need to be changed to the user id based on the web app
    user_id = "lead_001"
    
    while True:
        user_input = input("💬 Vous : ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 À bientôt !")
            break
        try:
            reply = agent_response(user_input, user_id=user_id)
            print(f"\n🧠 Agent :\n{reply}\n")
        except Exception as e:
            print(f"❌ Erreur : {e}")
