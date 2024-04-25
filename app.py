from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional,Input,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from PIL import Image
import numpy as np
from io import BytesIO
import requests
import pickle


app = Flask(__name__)

VOCABULARY_SIZE = 10000  # Example vocabulary size
MAX_LENGTH = 50  # Example maximum sequence length
NUM_DENSE_LAYERS=0  # Number of dense layers

def build_model(vocab_size, max_length, num_dense_layers):
    # Build a model with variable number of dense layers
    model = tf.keras.Sequential([
        Input(shape=(224, 224, 3)),  # Example input shape
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
    ])
    
    # Add Dense layers
    for _ in range(num_dense_layers):
        model.add(Dense(128, activation='relu'))
    
    model.add(Dense(vocab_size, activation='softmax'))
    
    return model

def load_model_weights(model, weights_file):
    model.load_weights(weights_file)

# Load the model
model = build_model(VOCABULARY_SIZE, MAX_LENGTH, NUM_DENSE_LAYERS)
load_model_weights(model, 'final.h5')

# Load the tokenizer
def load_tokenizer(tokenizer_file):
    with open(tokenizer_file, 'rb') as f:
        tokenizer_dict = pickle.load(f)
    
    tokenizer_class = getattr(tf.keras.layers, tokenizer_dict['class_name'])
    tokenizer = tokenizer_class.from_config(tokenizer_dict['config'])
    
    return tokenizer

# Note: You need to upload 'tokenizer.pkl' to Colab before running this code
tokenizer = load_tokenizer('tokenizer.pkl')

# Preprocess image
def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize to match model's expected input shape
    img = np.array(img)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Dummy function to generate captions (replace this with your actual code)
def generate_caption(image):
    # This is a placeholder function
    # You should replace this with your actual model prediction logic
    # For this example, it just returns a fixed caption
    return "A person riding a bicycle on a street"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_url' not in request.form:
        return jsonify({'error': 'No image URL provided'})

    image_url = request.form['image_url']
    
    try:
        img = preprocess_image(image_url)
    except Exception as e:
        return jsonify({'error': str(e)})

    # Generate caption using the loaded model
    caption = generate_caption(img)
    
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)