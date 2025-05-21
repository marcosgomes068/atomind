import sys
import os
import json
import io
import base64
from datetime import datetime
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env, se existir
load_dotenv()

app = Flask(__name__, static_folder='pag')
CORS(app)

# Diretório para salvar imagens geradas
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pag", "imagens_geradas")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pegar chaves de API das variáveis de ambiente (com fallback para valores vazios)
COHERE_API_KEY = os.environ.get('COHERE_API_KEY', '')
HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN', '')

@app.route('/api/perguntar', methods=['POST'])
def perguntar():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erro': 'Dados JSON não recebidos.', 'resposta': 'Erro ao processar sua pergunta.'}), 400
            
        pergunta = data.get('pergunta')
        if not pergunta:
            return jsonify({'erro': 'Pergunta não enviada.', 'resposta': 'Por favor, envie uma pergunta.'}), 400
        
        # Chamada direta à API da Cohere
        url = 'https://api.cohere.ai/v1/chat'
        headers = {
            'Authorization': f'Bearer {COHERE_API_KEY}',
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
            return jsonify({'resposta': data.get('text', 'Sem resposta da IA.')})
        else:
            app.logger.error(f'Erro na API Cohere: {response.status_code} - {response.text}')
            return jsonify({'resposta': f'Erro ao consultar a IA: {response.status_code}'}), 500
            
    except Exception as e:
        app.logger.error(f'Erro no servidor: {str(e)}')
        return jsonify({'resposta': f'Erro ao processar sua pergunta: {str(e)}'}), 500

@app.route('/')
def index():
    return send_from_directory('pag', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('pag', path)

@app.route('/api/gerar-imagem', methods=['POST'])
def gerar_imagem():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'erro': 'Dados JSON não recebidos.'}), 400
            
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'erro': 'Descrição da imagem não enviada.'}), 400
        
        # Melhoria do prompt para obter melhores resultados
        if not prompt.lower().startswith(("a photo of", "an image of")):
            prompt_melhorado = f"A detailed high quality image of {prompt}, 4k resolution, detailed"
        else:
            prompt_melhorado = prompt
        
        # Tentar vários modelos em caso de falha
        modelos = [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "prompthero/openjourney"
        ]
        
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        # Configuração dos parâmetros
        payload = {
            "inputs": prompt_melhorado,
            "parameters": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "negative_prompt": "blurry, bad anatomy, bad hands, cropped, worst quality, low quality, text, watermark"
            }
        }
        
        # Tentar cada modelo até obter sucesso
        for modelo in modelos:
            try:
                app.logger.info(f'Tentando gerar imagem com modelo: {modelo}')
                api_url = f"https://api-inference.huggingface.co/models/{modelo}"
                
                # Chamada à API do Hugging Face
                response = requests.post(api_url, headers=headers, json=payload, timeout=20)
                
                # Verificar se é um erro de modelo carregando
                if response.status_code != 200:
                    if "loading" in response.text.lower():
                        app.logger.info(f'Modelo {modelo} ainda carregando, tentando próximo modelo...')
                        continue
                    
                    app.logger.warning(f'Erro no modelo {modelo}: {response.status_code} - {response.text}')
                    continue  # Tentar o próximo modelo
                
                # Se chegou aqui, temos uma resposta bem-sucedida
                # Processar a imagem recebida do Hugging Face
                try:
                    # Verificar se o conteúdo é uma imagem válida
                    from PIL import Image
                    image = Image.open(io.BytesIO(response.content))
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"img_{timestamp}.png"
                    img_path = os.path.join(OUTPUT_DIR, filename)
                    
                    # Salvar a imagem
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    
                    app.logger.info(f'Imagem gerada com sucesso usando modelo {modelo}')
                    # Retornar o caminho para o arquivo
                    return jsonify({'imagem_url': f"/imagens_geradas/{filename}"})
                    
                except Exception as img_error:
                    app.logger.error(f'Erro ao processar imagem do modelo {modelo}: {str(img_error)}')
                    continue  # Tentar o próximo modelo
            
            except requests.RequestException as req_error:
                app.logger.error(f'Erro de requisição no modelo {modelo}: {str(req_error)}')
                continue  # Tentar o próximo modelo
        
        # Se chegou aqui, nenhum dos modelos funcionou
        # Usar fallback para Unsplash
        app.logger.warning('Todos os modelos falharam, usando fallback Unsplash')
        unsplash_url = f"https://source.unsplash.com/800x500/?{prompt.replace(' ', ',')}"
        return jsonify({'imagem_url': unsplash_url, 'fallback': True})
            
    except Exception as e:
        app.logger.error(f'Erro ao gerar imagem: {str(e)}')
        return jsonify({'erro': f'Erro ao gerar imagem: {str(e)}'}), 500

# Rota para servir as imagens geradas
@app.route('/imagens_geradas/<path:filename>')
def serve_generated_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Em desenvolvimento usamos modo debug
    # Em produção, usamos o host 0.0.0.0 para escutar em todas as interfaces
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
