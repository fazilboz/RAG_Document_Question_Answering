from ollama import chat
import uuid

class Chatbot:
    def __init__(self, vectorstore, api_key=None):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())

    def respond(self, user_message: str):
        retrieved_docs = self.vectorstore.retrieve(user_message)

        context = "\n\n".join(
            [doc["text"] for doc in retrieved_docs]
        )

        prompt = f"""
Use the provided context to answer the question.

Context:
{context}

Question:
{user_message}
"""

        response = chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"], retrieved_docs
	
