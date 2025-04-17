# Text Emotion Recognition API

This project consists of a RESTful API built with Flask that uses a machine learning model to predict the emotion present in a given text. The model was trained using the TensorFlow/Keras library.

## Overview

The API receives a POST request to the `/predict` endpoint with text in JSON format and returns the predicted emotion (currently, 'positive' or 'negative').

## Prerequisites

* Python 3.6 or higher
* Pip (Python package installer)
* Git (to clone the repository)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <Your repository URL from GitHub>
    cd emotion-recognition-api
    ```

    (Replace `<Your repository URL from GitHub>` with the actual URL of your repository).

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## How to Use the API

1.  **Run the Flask server:**

    ```bash
    python -m src.api
    ```

    This will start the Flask server locally (usually at `http://127.0.0.1:5000/`).

2.  **Send a POST request to the `/predict` endpoint:**

    You can use tools like `curl`, Postman, or a Python script to send requests.

    **Example using `curl` in the terminal (Bash):**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"text": "I am very happy today!"}' [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
    ```

    **Example using `curl` in the terminal (PowerShell):**

    ```powershell
    Invoke-WebRequest -Uri [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) -Method Post -Headers @{'Content-Type'='application/json'} -Body '{"text": "I am very happy today!"}'
    ```

    The response will be a JSON containing the predicted emotion:

    ```json
    {
      "emotion": "positive"
    }
    ```

## Model Training

The machine learning model used in this API was trained with the Sentiment140 dataset. The code to train the model is in the `src/model.py` file.

To train the model (or re-train it), follow these steps:

1.  **Make sure you have the Sentiment140 dataset** (the `training.1600000.processed.noemoticon.csv` file) in the `data/` folder. You can download it from sources like Kaggle.
2.  **Run the training script:**

    ```bash
    python -m src.model
    ```

    This script will load the data, preprocess it, train the model, and save the trained model to `models/emotion_recognition_model.h5` and the tokenizer to `models/tokenizer.pkl`.

## API Endpoints

* `/predict` (POST): Receives a JSON with the key `text` and returns a JSON with the key `emotion` (the predicted emotion).

## Future Improvements (Optional)

* Improve the model's accuracy.
* Add support for more emotions.
* Implement unit tests.
* Create a frontend to interact with the API.
* Explore different model architectures.

## Author (Optional)

Your Name or GitHub Username