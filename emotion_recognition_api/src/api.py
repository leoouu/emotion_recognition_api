from flask import Flask, request, jsonify
from src.model import predict_emotion

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Por favor, forneça um texto para análise.'}), 400

    text = data['text']
    emotion = predict_emotion(text)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)