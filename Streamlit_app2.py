import streamlit as st
import uuid
from Agent.agent2005 import agent_response

# === Config Streamlit ===
st.set_page_config(
    page_title="CCI Mexico AI Assistant",
    layout="centered"
)

# === Initialisation de session ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# === UI ===
st.title("Assistant CCI Mexico 🇲🇽🤖")

st.markdown("""
Bienvenue ! Je suis l'assistant virtuel de la Chambre de Commerce et d'Industrie Franco-mexicaine.
**Comment puis-je vous aider aujourd’hui ?**

<br>

¡Bienvenido! Soy el asistente virtual de la Cámara de Comercio e Industria Franco-Mexicana.  
¿En qué puedo ayudarle hoy?
""", unsafe_allow_html=True)

# === Historique affiché ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Zone de saisie ===
if prompt := st.chat_input("Votre message..."):
    # Ajouter message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_response(prompt, st.session_state.session_id)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

