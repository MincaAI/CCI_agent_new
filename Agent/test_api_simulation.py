import asyncio
from agent06redis import agent_response

# Simule des messages envoyés avec un chat_id identique
conversation = [
    {"chat_id": "33662993596", "message": "Bonjour"},
    {"chat_id": "33662993596", "message": "J'aimerais en savoir plus sur les services d'implantation"},
    {"chat_id": "33662993596", "message": "Est-ce que vous aidez à créer une entreprise ?"},
    {"chat_id": "33662993596", "message": "Combien ça coûte ?"}
]

async def run_test():
    for i, item in enumerate(conversation, 1):
        chat_id = item["chat_id"]
        user_input = item["message"]

        print(f"\n🟦 Message {i}: {user_input} (chat_id: {chat_id})")
        response = await agent_response(user_input=user_input, chat_id=chat_id)
        print(f"🟩 Réponse {i}: {response}")

asyncio.run(run_test())