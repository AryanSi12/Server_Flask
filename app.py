from flask import Flask, request, jsonify
from flask_cors import CORS ,cross_origin
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

# Disable GPU for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize Flask app
app = Flask(__name__)

# Enable CORS globally (Allow all origins, but you can change "*" to a specific frontend URL)
CORS(app, supports_credentials=True)

# Load trained model
model_path = "my_model1.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Load tokenizer
tokenizer_path = "lstm_tokenizer.joblib"
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

tokenizer = joblib.load(tokenizer_path)

# Tokenizer parameters
max_len = 100

@app.route('/predict', methods=['OPTIONS', 'POST'])
@cross_origin()
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight passed'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 200

    try:
        # Get JSON data from request
        data = request.get_json()
        comments = data.get('comments', [])

        if not comments or not isinstance(comments, list):
            response = jsonify({'error': 'Invalid input. Provide a list of comments.'})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 400

        # Preprocess comments
        sequences = tokenizer.texts_to_sequences(comments)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Predict sentiment
        predictions = model.predict(padded_sequences)

        sentiment_labels = ['negative', 'neutral', 'positive']
        results = [{'comment': comment, 'sentiment': sentiment_labels[np.argmax(pred)]}
                   for comment, pred in zip(comments, predictions)]

        # Return JSON response with proper CORS headers
        response = jsonify({'predictions': results})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response
    except Exception as e:
        response = jsonify({'error': f"An error occurred: {str(e)}"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port)
