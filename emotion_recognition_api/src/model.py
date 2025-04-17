import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

try:
    with open('models/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    tokenizer = None
    print("Aviso: tokenizer.pkl não encontrado. Certifique-se de treinar o modelo primeiro.")

# Carregar o modelo treinado
try:
     emotion_model = tf.keras.models.load_model('models/emotion_recognition_model.h5')
except FileNotFoundError:
     emotion_model = None
     print("Erro: emotion_recognition_model.h5 não encontrado. Certifique-se de treinar o modelo primeiro.")
emotion_model = None # Inicializar emotion_model como None

def predict_emotion(text):
    global emotion_model
    if emotion_model is None:
        try:
            emotion_model = tf.keras.models.load_model('models/emotion_recognition_model.h5')
        except FileNotFoundError:
            return "Modelo não encontrado. Execute o treinamento primeiro."
    if tokenizer is None:
        return "Tokenizador não encontrado. Execute o treinamento primeiro."

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post') 

    prediction = emotion_model.predict(padded_sequences)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

    emotion_labels = {0: 'negative', 1: 'positive'} 
    predicted_emotion = emotion_labels.get(predicted_class, 'desconhecida')

    return predicted_emotion

# AQUI começa o treinamento 
if __name__ == '__main__':
    # Carregamento dos Dados 
    file_path = 'data/training.1600000.processed.noemoticon.csv'
    try:
        data = pd.read_csv(file_path, encoding='latin1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])
        data = data[data['target'].isin([0, 4])]
        sentiment_labels = {0: 'negative', 4: 'positive'}
        data['emotion'] = data['target'].map(sentiment_labels)
        textos = data['text'].values
        emocoes = data['emotion'].values
        print(f"Número de exemplos carregados: {len(textos)}")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}. Certifique-se de que o arquivo existe.")
        exit()

    # Pré-processamento do Texto
    vocab_size = 10000
    max_length = 100
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
    tokenizer.fit_on_texts(textos)
    sequences = tokenizer.texts_to_sequences(textos)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post', padding='post')

    # Salvar o tokenizador
    with open('models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizador salvo em models/tokenizer.pkl")


    # Codificação das Emoçes
    label_encoder = LabelEncoder()
    emocoes_encoded = label_encoder.fit_transform(emocoes)
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"Mapeamento de emoções: {label_mapping}")
    emocoes_encoded = np.array(emocoes_encoded)


    # Divisão dos Dados
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, emocoes_encoded, test_size=0.2, random_state=42)



    # Construção do Modelo
    embedding_dim = 16
    lstm_units = 32
    num_classes = len(label_encoder.classes_)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(lstm_units),
        Dense(num_classes, activation='softmax')
    ])



    # Compilação do Modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinamento do Modelo
    epochs = 10
    batch_size = 32
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Avaliação do Modelo
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Acurácia nos dados de teste: {accuracy:.4f}')

    # Salvar o Modelo
    model.save('models/emotion_recognition_model.h5')
    print("Modelo treinado e salvo em models/emotion_recognition_model.h5")