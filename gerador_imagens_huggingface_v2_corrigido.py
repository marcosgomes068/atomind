#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gerador de Imagens com Hugging Face Inference API

Este script permite gerar imagens de duas formas diferentes:
1. Text-to-Image: Gerar imagens a partir de descrições textuais
2. Image-to-Image: Modificar imagens existentes com base em prompts

Autor: GitHub Copilot
Data: Maio de 2025
"""

# Importação das bibliotecas
import requests
import json
import os
import io
import base64
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# Suprimir os avisos para deixar a execução mais limpa
warnings.filterwarnings('ignore')

# Configurar o matplotlib para exibir imagens maiores
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['figure.dpi'] = 100

print("Todas as bibliotecas foram importadas com sucesso!")

# Configuração do Token da Hugging Face
# O token deve começar com "hf_" - substitua pelo seu token real
# Para obter um token, acesse: https://huggingface.co/settings/tokens
import os
from dotenv import load_dotenv
# Carregar variáveis de ambiente
load_dotenv()
API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN', '')  # Obter token da variável de ambiente

# Criando uma pasta para salvar as imagens geradas
OUTPUT_DIR = "imagens_geradas"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Pasta {OUTPUT_DIR} criada/verificada com sucesso!")

# Verificação simples do token
if not API_TOKEN.startswith("hf_"):
    print("⚠️ ATENÇÃO: O token da Hugging Face parece incorreto. Deve começar com 'hf_'.")
else:
    print("✅ Token configurado. Pronto para usar a API!")

# Configuração dos modelos - você pode alternar entre diferentes modelos disponíveis
MODELOS = {
    "text2image": [
        "runwayml/stable-diffusion-v1-5",      # Modelo popular e bem estabelecido
        "stabilityai/stable-diffusion-xl-base-1.0",  # SDXL Base
        "prompthero/openjourney"               # Estilo Midjourney
    ],
    "image2image": [
        "runwayml/stable-diffusion-v1-5",      # Também funciona bem para image-to-image
        "timbrooks/instruct-pix2pix",          # Bom para edição de imagens com instruções
    ]
}

# Modelo padrão para text-to-image
MODELO_PADRAO_TEXT2IMAGE = MODELOS["text2image"][0]
# Modelo padrão para image-to-image
MODELO_PADRAO_IMAGE2IMAGE = MODELOS["image2image"][0]

def verificar_token_e_conexao():
    """
    Verifica se o token da Hugging Face está configurado e se é possível conectar à API
    
    Retorna:
    - bool: True se estiver tudo ok, False caso contrário
    """
    global MODELO_PADRAO_TEXT2IMAGE, MODELO_PADRAO_IMAGE2IMAGE
    
    # Verificar formato básico do token
    if not API_TOKEN.startswith("hf_"):
        print("❌ Erro: O token da Hugging Face está em formato incorreto.")
        print("   O token deve começar com 'hf_'. Verifique seu token na página da Hugging Face.")
        return False
    
    # Tentar uma requisição simples para verificar a conexão e o token
    try:
        print("Verificando conexão com a API da Hugging Face...")
        
        # URL do modelo de teste
        api_url = f"https://api-inference.huggingface.co/models/{MODELO_PADRAO_TEXT2IMAGE}"
        
        # Headers com o token
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        
        # Fazer uma requisição GET simples
        response = requests.get(api_url, headers=headers)
        
        # Verificar o código de status
        if response.status_code == 401 or response.status_code == 403:
            print("❌ Erro: Token inválido ou expirado.")
            print("   Por favor, verifique seu token na página da Hugging Face.")
            return False
        elif response.status_code >= 500:
            print("⚠️ Aviso: Os servidores da Hugging Face podem estar sobrecarregados ou indisponíveis.")
            print("   Mas vamos tentar prosseguir mesmo assim...")
            return True
        elif response.status_code == 404:
            print(f"⚠️ Aviso: Modelo {MODELO_PADRAO_TEXT2IMAGE} não encontrado (404).")
            print("   Tentando com modelos alternativos...")
            
            # Tentar com o segundo modelo da lista
            alt_api_url = f"https://api-inference.huggingface.co/models/{MODELOS['text2image'][1]}"
            alt_response = requests.get(alt_api_url, headers=headers)
            
            if alt_response.status_code == 200:
                print(f"✅ Conexão com o modelo alternativo {MODELOS['text2image'][1]} estabelecida!")
                # Definir o modelo alternativo como padrão
                MODELO_PADRAO_TEXT2IMAGE = MODELOS['text2image'][1]
                return True
            else:
                print(f"⚠️ Aviso: Também recebido código de status {alt_response.status_code} para o modelo alternativo.")
                print("   Prosseguindo, mas pode haver problemas com a API.")
                return True
        else:
            print("✅ Conexão estabelecida com sucesso! O token parece estar funcionando corretamente.")
            return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Erro: Não foi possível conectar aos servidores da Hugging Face.")
        print("   Verifique sua conexão com a internet.")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {str(e)}")
        return True  # Retornamos True mesmo com erro para permitir tentar usar a API

def gerar_por_texto(prompt, num_inference_steps=30, guidance_scale=7.5, save=True, modelo=None):
    """
    Gera uma imagem a partir de um prompt de texto usando o modelo Stable Diffusion
    
    Parâmetros:
    - prompt (str): Descrição textual da imagem que você deseja gerar
    - num_inference_steps (int): Número de etapas de inferência (mais etapas = mais detalhes, mas mais lento)
    - guidance_scale (float): Quão fielmente o modelo deve seguir o prompt (valores maiores = mais fidelidade)
    - save (bool): Se True, salva a imagem gerada no disco
    - modelo (str): Modelo específico a ser usado, se None usa o modelo padrão
    
    Retorna:
    - Objeto de imagem PIL
    """
    # Adicionar melhores práticas para prompts
    prompt_melhorado = prompt
    if not prompt.lower().startswith(("a photo of", "an image of")):
        # Melhorar um pouco o prompt para obter melhores resultados
        prompt_melhorado = f"A detailed high quality image of {prompt}, 4k resolution, detailed"
    try:
        # Usar o modelo especificado ou o padrão
        modelo_atual = modelo if modelo else MODELO_PADRAO_TEXT2IMAGE
        
        # 1. Configurar a URL do modelo e headers para autenticação
        api_url = f"https://api-inference.huggingface.co/models/{modelo_atual}"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
          # 2. Preparar o payload da requisição com os parâmetros
        payload = {
            "inputs": prompt_melhorado,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": "blurry, bad anatomy, bad hands, cropped, worst quality, low quality, text, watermark"
            }
        }
          # 3. Fazer a requisição POST para a API
        print(f"Gerando imagem para o prompt: '{prompt}'...")
        if prompt_melhorado != prompt:
            print(f"Prompt melhorado: '{prompt_melhorado}'")
        print(f"Usando modelo: {modelo_atual}")
        print(f"Passos de inferência: {num_inference_steps} | Guidance scale: {guidance_scale}")
        print("Isso pode levar alguns minutos. Por favor, aguarde...")
        
        response = requests.post(api_url, headers=headers, json=payload)
          # 4. Verificar se a requisição foi bem-sucedida
        if response.status_code != 200:
            print(f"Erro na requisição: {response.status_code}")
            print(f"Mensagem: {response.text}")
            
            # Verificar se o erro é devido ao modelo estar carregando
            if "loading" in response.text.lower() or "still loading" in response.text.lower():
                print("O modelo está sendo carregado pela primeira vez. Tentando novamente em 10 segundos...")
                import time
                time.sleep(10)  # Esperar 10 segundos antes de tentar novamente
                # Tentar a mesma requisição novamente
                print("Tentando novamente...")
                response = requests.post(api_url, headers=headers, json=payload)
                # Verificar se a segunda tentativa foi bem-sucedida
                if response.status_code == 200:
                    print("Segunda tentativa bem-sucedida!")
                else:
                    print(f"Segunda tentativa falhou: {response.status_code}")
                    print(f"Mensagem: {response.text}")
            
            # Se o modelo atual falhar e não for o último da lista, tente o próximo
            if modelo_atual != MODELOS["text2image"][-1]:
                proximo_indice = MODELOS["text2image"].index(modelo_atual) + 1
                if proximo_indice < len(MODELOS["text2image"]):
                    proximo_modelo = MODELOS["text2image"][proximo_indice]
                    print(f"Tentando com modelo alternativo: {proximo_modelo}")
                    return gerar_por_texto(prompt, num_inference_steps, guidance_scale, save, proximo_modelo)
            return None
            
        # 5. Transformar a resposta em uma imagem
        image = Image.open(io.BytesIO(response.content))
        
        # 6. Salvar a imagem se solicitado
        if save:
            # Criar um nome de arquivo baseado no prompt e timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Limitar o tamanho do nome do arquivo e remover caracteres especiais
            prompt_slug = "".join(c for c in prompt if c.isalnum() or c in " ")[:30].strip().replace(" ", "_")
            filename = f"{OUTPUT_DIR}/text_{prompt_slug}_{timestamp}.png"
            image.save(filename)
            print(f"Imagem salva como: {filename}")
        
        # 7. Exibir a imagem
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')  # Remove os eixos
        plt.title(f"Prompt: {prompt}\nModelo: {modelo_atual}")
        plt.show()
        
        return image
    
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")
        return None

def gerar_por_imagem(prompt, caminho_imagem, strength=0.8, num_inference_steps=30, guidance_scale=7.5, save=True, modelo=None):
    """
    Modifica uma imagem existente com base em um prompt usando o modelo Stable Diffusion
    
    Parâmetros:
    - prompt (str): Descrição textual das modificações desejadas
    - caminho_imagem (str): Caminho para a imagem de entrada
    - strength (float): Intensidade da transformação (0.0 a 1.0, onde 1.0 = mudança completa)
    - num_inference_steps (int): Número de etapas de inferência
    - guidance_scale (float): Quão fielmente seguir o prompt
    - save (bool): Se True, salva a imagem gerada no disco
    - modelo (str): Modelo específico a ser usado, se None usa o modelo padrão
    
    Retorna:
    - Objeto de imagem PIL
    """
    try:
        # Usar o modelo especificado ou o padrão
        modelo_atual = modelo if modelo else MODELO_PADRAO_IMAGE2IMAGE
        
        # 1. Verificar se o arquivo existe
        if not os.path.exists(caminho_imagem):
            print(f"Erro: O arquivo '{caminho_imagem}' não existe.")
            return None
        
        # 2. Abrir e redimensionar a imagem (o modelo geralmente espera tamanhos específicos)
        image = Image.open(caminho_imagem)
        # Redimensionar para um tamanho adequado para o modelo (múltiplo de 8)
        width, height = image.size
        # Ajustar para dimensões que sejam múltiplos de 8
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        if new_width != width or new_height != height:
            image = image.resize((new_width, new_height))
            print(f"Imagem redimensionada de {width}x{height} para {new_width}x{new_height}")
        
        # 3. Converter a imagem para formato base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 4. Configurar URL e headers
        api_url = f"https://api-inference.huggingface.co/models/{modelo_atual}"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        
        # 5. Preparar o payload da requisição
        payload = {
            "inputs": {
                "prompt": prompt,
                "image": img_str,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
        }
        
        # 6. Enviar a requisição
        print(f"Modificando imagem com o prompt: '{prompt}'...")
        print(f"Usando modelo: {modelo_atual}")
        print(f"Intensidade da transformação: {strength * 100}%")
        print(f"Passos de inferência: {num_inference_steps} | Guidance scale: {guidance_scale}")
        print("Isso pode levar alguns minutos. Por favor, aguarde...")
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        # 7. Verificar resposta
        if response.status_code != 200:
            print(f"Erro na requisição: {response.status_code}")
            print(f"Mensagem: {response.text}")
            
            # Se o modelo atual falhar e não for o último da lista, tente o próximo
            if modelo_atual != MODELOS["image2image"][-1]:
                proximo_indice = MODELOS["image2image"].index(modelo_atual) + 1
                if proximo_indice < len(MODELOS["image2image"]):
                    proximo_modelo = MODELOS["image2image"][proximo_indice]
                    print(f"Tentando com modelo alternativo: {proximo_modelo}")
                    return gerar_por_imagem(prompt, caminho_imagem, strength, num_inference_steps, 
                                           guidance_scale, save, proximo_modelo)
            return None
            
        # 8. Transformar a resposta em uma imagem
        result_image = Image.open(io.BytesIO(response.content))
        
        # 9. Salvar a imagem se solicitado
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Criar um nome de arquivo baseado no prompt e nome do arquivo original
            base_filename = os.path.basename(caminho_imagem).split(".")[0]
            prompt_slug = "".join(c for c in prompt if c.isalnum() or c in " ")[:20].strip().replace(" ", "_")
            filename = f"{OUTPUT_DIR}/img2img_{base_filename}_{prompt_slug}_{timestamp}.png"
            result_image.save(filename)
            print(f"Imagem salva como: {filename}")
        
        # 10. Exibir a imagem original e a modificada lado a lado
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Imagem Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        plt.title(f"Imagem Modificada\nPrompt: {prompt}\nModelo: {modelo_atual}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return result_image
    
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")
        return None

def menu_modo_linha_comando():
    """
    Função para execução em modo de linha de comando com um menu interativo
    """
    # Verificar o token e conexão primeiro
    if not verificar_token_e_conexao():
        print("\n❌ Configurações iniciais incompletas. Revise seu token e conexão.")
        return

    while True:
        print("\n" + "="*50)
        print("🎨 GERADOR DE IMAGENS COM HUGGING FACE API 🎨")
        print("="*50)
        print("1. Gerar imagem a partir de texto (Text-to-Image)")
        print("2. Modificar imagem existente (Image-to-Image)")
        print("3. Exibir modelos disponíveis")
        print("4. Alterar token da API")
        print("5. Sair")
        
        escolha = input("\nEscolha uma opção (1-5): ").strip()
        
        if escolha == "1":
            # Text-to-Image
            prompt = input("\nDigite a descrição da imagem que você deseja gerar: ").strip()
            if not prompt:
                print("É necessário fornecer uma descrição.")
                continue
                
            # Parâmetros opcionais
            try:
                passos = int(input("Número de passos de inferência (padrão: 30): ").strip() or "30")
                guidance = float(input("Guidance scale (1 a 20, padrão: 7.5): ").strip() or "7.5")
                
                # Escolher o modelo
                print("\nModelos disponíveis para Text-to-Image:")
                for i, modelo in enumerate(MODELOS["text2image"]):
                    if modelo == MODELO_PADRAO_TEXT2IMAGE:
                        print(f"{i+1}. {modelo} (PADRÃO)")
                    else:
                        print(f"{i+1}. {modelo}")
                
                escolha_modelo = int(input(f"\nEscolha um modelo (1-{len(MODELOS['text2image'])}, ou 0 para usar o padrão): ").strip() or "0")
                
                if escolha_modelo > 0 and escolha_modelo <= len(MODELOS["text2image"]):
                    modelo_selecionado = MODELOS["text2image"][escolha_modelo-1]
                else:
                    modelo_selecionado = MODELO_PADRAO_TEXT2IMAGE
                    
                print(f"\nGerando imagem... (Modelo: {modelo_selecionado})")
                gerar_por_texto(prompt, passos, guidance, True, modelo_selecionado)
            except ValueError:
                print("Erro: Por favor insira valores numéricos válidos para os parâmetros.")
        
        elif escolha == "2":
            # Image-to-Image
            caminho = input("\nDigite o caminho completo para a imagem que deseja modificar: ").strip()
            if not os.path.exists(caminho):
                print(f"Erro: O arquivo '{caminho}' não existe.")
                continue
                
            prompt = input("Digite a descrição das modificações desejadas: ").strip()
            if not prompt:
                print("É necessário fornecer uma descrição para as modificações.")
                continue
                
            # Parâmetros opcionais
            try:
                strength = float(input("Intensidade da transformação (0.1 a 1.0, padrão: 0.8): ").strip() or "0.8")
                strength = max(0.1, min(1.0, strength))  # Limitar entre 0.1 e 1.0
                
                passos = int(input("Número de passos de inferência (padrão: 30): ").strip() or "30")
                guidance = float(input("Guidance scale (1 a 20, padrão: 7.5): ").strip() or "7.5")
                
                # Escolher o modelo
                print("\nModelos disponíveis para Image-to-Image:")
                for i, modelo in enumerate(MODELOS["image2image"]):
                    if modelo == MODELO_PADRAO_IMAGE2IMAGE:
                        print(f"{i+1}. {modelo} (PADRÃO)")
                    else:
                        print(f"{i+1}. {modelo}")
                
                escolha_modelo = int(input(f"\nEscolha um modelo (1-{len(MODELOS['image2image'])}, ou 0 para usar o padrão): ").strip() or "0")
                
                if escolha_modelo > 0 and escolha_modelo <= len(MODELOS["image2image"]):
                    modelo_selecionado = MODELOS["image2image"][escolha_modelo-1]
                else:
                    modelo_selecionado = MODELO_PADRAO_IMAGE2IMAGE
                
                print(f"\nModificando imagem... (Modelo: {modelo_selecionado})")
                gerar_por_imagem(prompt, caminho, strength, passos, guidance, True, modelo_selecionado)
            except ValueError:
                print("Erro: Por favor insira valores numéricos válidos para os parâmetros.")
        
        elif escolha == "3":
            # Exibir modelos disponíveis
            print("\nModelos Text-to-Image disponíveis:")
            for i, modelo in enumerate(MODELOS["text2image"]):
                if modelo == MODELO_PADRAO_TEXT2IMAGE:
                    print(f"{i+1}. {modelo} (PADRÃO)")
                else:
                    print(f"{i+1}. {modelo}")
                
            print("\nModelos Image-to-Image disponíveis:")
            for i, modelo in enumerate(MODELOS["image2image"]):
                if modelo == MODELO_PADRAO_IMAGE2IMAGE:
                    print(f"{i+1}. {modelo} (PADRÃO)")
                else:
                    print(f"{i+1}. {modelo}")
            
            input("\nPressione Enter para continuar...")
            
        elif escolha == "4":
            # Alterar token da API
            global API_TOKEN
            novo_token = input("\nDigite seu novo token da Hugging Face (deve começar com 'hf_'): ").strip()
            
            if not novo_token.startswith("hf_"):
                print("Token inválido! Deve começar com 'hf_'")
                continue
                
            API_TOKEN = novo_token
            print("Token atualizado com sucesso!")
            
            # Verificar o novo token
            verificar_token_e_conexao()
            
        elif escolha == "5":
            # Sair
            print("\nObrigado por usar o Gerador de Imagens com Hugging Face API!")
            print("Desenvolvido com 💻 e ❤️ pelo GitHub Copilot")
            break
            
        else:
            print("Opção inválida! Por favor, escolha uma opção entre 1 e 5.")

# Se o script for executado diretamente (não importado como módulo)
if __name__ == "__main__":
    try:
        print("\nIniciando Gerador de Imagens com Hugging Face API...")
        menu_modo_linha_comando()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário. Encerrando...")
    except Exception as e:
        print(f"\nErro inesperado: {str(e)}")
    finally:
        print("\nObrigado por usar o Gerador de Imagens! Até a próxima.")
