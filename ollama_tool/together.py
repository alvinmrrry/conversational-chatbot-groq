import os
from together import Together

client = Together(api_key='bc70fd3d4726928fdd8fe258272fb92ab4f518ba330079c9ecbcd61f4e866cb1')

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    messages=[
        {
                "role": "user",
                "content": "hi"
        },
        {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
        }
],
    max_tokens=512,
    temperature=0.7,
    top_p=0.7,
    top_k=50,
    repetition_penalty=1,
    stop=["<|eot_id|>","<|eom_id|>"],
    stream=True
)
print(response.choices[0].message.content)