# Atomai - Assistente Virtual com Geração de Imagens

Uma aplicação web que combina chat com IA e geração de imagens usando modelos de linguagem e difusão.

## Funcionalidades

- Chat interativo com IA usando Cohere API
- Geração de imagens a partir de descrições textuais usando Hugging Face
- Interface responsiva e moderna
- Reconhecimento de voz (navegadores compatíveis)

## Tecnologias

- Backend: Flask (Python)
- Frontend: HTML/CSS/JavaScript
- APIs: Cohere, Hugging Face

## Instalação Local

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Configure as variáveis de ambiente no arquivo `.env`
4. Execute o servidor: `python app.py`
5. Acesse: http://localhost:5000

## Variáveis de Ambiente

- `COHERE_API_KEY`: Chave de API da Cohere
- `HUGGINGFACE_API_TOKEN`: Token da Hugging Face
- `FLASK_ENV`: Ambiente (development/production)

## Deploy

Esta aplicação está configurada para deploy em plataformas como Render, Heroku ou Railway.
