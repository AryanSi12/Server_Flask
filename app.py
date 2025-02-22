from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

# ===============================
# 🔧 Environment & Performance Fixes
# ===============================
# Disable GPU for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Optimize TensorFlow settings
tf.compat.v1.disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ===============================
# 🚀 Initialize Flask App
# ===============================
app = Flask(__name__)

# Define allowed frontend origin (Modify this if needed)
FRONTEND_URL = "https://comment-analyzer-frontend.vercel.app"

# Enable CORS for specific endpoints
CORS(app, resources={r"/predict": {"origins": FRONTEND_URL}}, supports_credentials=True)

# ===============================
# 🔥 Load Model & Tokenizer (Only Once)
# ===============================
model_path = "my_model1.keras"
tokenizer_path = "lstm_tokenizer.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

# Load trained model
model = tf.keras.models.load_model(model_path)

# Load tokenizer
tokenizer = joblib.load(tokenizer_path)

# Tokenizer parameters
max_len = 100

# ===============================
# 📌 API Endpoint: /predict
# ===============================
@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    # ✅ Handle CORS Preflight (OPTIONS request)
    if request.method == 'OPTIONS':
        return cors_response({'message': 'CORS preflight passed'}, 200)

    try:
        # ✅ Parse JSON request
        data = request.get_json()
        comments = data.get('comments', [])

        # ✅ Validate Input
        if not comments or not isinstance(comments, list):
            return cors_response({'error': 'Invalid input. Provide a list of comments.'}, 400)

        # ✅ Preprocess comments
        sequences = tokenizer.texts_to_sequences(comments)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

        # ✅ Predict sentiment
        predictions = model.predict(padded_sequences)

        sentiment_labels = ['negative', 'neutral', 'positive']
        results = [{'comment': comment, 'sentiment': sentiment_labels[np.argmax(pred)]}
                   for comment, pred in zip(comments, predictions)]

        return cors_response({'predictions': results})

    except Exception as e:
        return cors_response({'error': f"An error occurred: {str(e)}"}, 500)


# ===============================
# 🛠️ Helper Function for CORS Responses
# ===============================
def cors_response(data, status=200):
    """Wrap JSON response with proper CORS headers"""
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", FRONTEND_URL)
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, status


# ===============================
# 🔥 Run Flask App
# ===============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5500))
    app.run(host="0.0.0.0", port=port)
