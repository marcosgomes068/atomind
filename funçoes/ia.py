import requests

def responder_cohere(pergunta, api_key):
    url = 'https://api.cohere.ai/v1/chat'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'message': pergunta,
        'model': 'command-r-plus',
        'chat_history': []
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('text', 'Sem resposta da IA.')
    else:
        return f'Erro: {response.status_code} - {response.text}'
