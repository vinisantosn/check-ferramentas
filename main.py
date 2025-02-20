import os

from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import base64
from io import BytesIO
from ultralytics import YOLO

# Carregar o modelo treinado
MODEL_PATH = "best.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)
model.to(device)

# Lista de classes
CLASSES = [
    "alicateBico", "alicateCorte", "alicateCripadorAm", "alicateCripadorAz", "alicateUniversal",
    "chave01", "chave02", "chave03", "chave04", "chave05", "chave06",
    "crimpadorFemea", "desecapador", "estilete", "etiquetadora", "fitaMetrica",
    "idCabo", "multimetro", "parafusadeira", "pincel", "tesoura", "testeCabo", "testeDeFonte"
]

# Configuração do Flask
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture():
    data = request.json.get("image")
    if not data:
        return jsonify({"error": "Nenhuma imagem recebida"}), 400

    # Decodificar a imagem base64
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Processar imagem
    results = model(img)[0]  # Obter os resultados da primeira saída

    # Criar tabela de presença das ferramentas
    presence_table = {tool: 0 for tool in CLASSES}

    for box in results.boxes.data.tolist():  # Acessar detecções
        class_id = int(box[5])  # Índice da classe detectada
        if class_id < len(CLASSES):
            class_name = CLASSES[class_id]
            presence_table[class_name] = 1

    return jsonify({"results": presence_table})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
