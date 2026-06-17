from groq import Groq

def synthesize_answer(query, context):
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": f"Query: {query}\nContext: {context}"}]
    )
    return response.choices[0].message.content
