from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from TTS.api import TTS
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json
from flask_cors import CORS

# Load the pre-trained emotion detection model
model = load_model('C:/LexiPal - AI/emotion_detection_model.keras')

# Load the pre-trained TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Tokenizer setup (ensure you have the tokenizer set up correctly with the same data you trained on)
with open("tokenizer.json", "r") as file:
    tokenizer_json = json.load(file)
tokenizer = tokenizer_from_json(tokenizer_json)
# Load or train your tokenizer with actual texts (training_texts should be actual data used for training)
# Example: tokenizer.fit_on_texts(training_texts)

label_dict = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}  

# Ensure the output folder exists
if not os.path.exists('static/audio'):
    os.makedirs('static/audio')

# Emotion Prediction Function using TensorFlow model
def predict_emotion(text):
    # Ensure the text is not empty or None
    if not text or not text.strip():
        raise ValueError("Input text is empty or invalid")

    # Tokenizing and padding text
    sequence = tokenizer.texts_to_sequences([text])
    if not sequence or not sequence[0]:
        raise ValueError("Text could not be tokenized into a valid sequence")

    padded = pad_sequences(sequence, maxlen=100)

    # Predicting emotion using the trained model
    prediction = model.predict(padded)

    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(prediction)

    # Reverse the label_dict to map index to emotion label
    emotion = [emotion for emotion, index in label_dict.items() if index == predicted_class_index][0]

    return emotion

# Generate speech based on detected emotion
def generate_speech(text, emotion, speaker_wav=None):
    try:
        emotion_text = f"{text}"
        output_file = f"static/audio/output_{emotion}.wav"

        if speaker_wav:
            tts.tts_to_file(text=emotion_text, speaker_wav=speaker_wav, file_path=output_file)
        else:
            tts.tts_to_file(text=emotion_text, file_path=output_file)

        print(f"Generated speech with emotion '{emotion}' saved as '{output_file}'.")
        return output_file
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

# API endpoint to process text and generate speech
@app.route('/generate_speech', methods=['POST'])
def generate_emotional_speech():
    data = request.get_json()
    text = data.get('text')
    if not text or not text.strip():
        return jsonify({"error": "Text is required and cannot be empty"}), 400

    print(f"Received text: {text}")
    try:
        # Step 1: Detect emotion using your TensorFlow model
        detected_emotion = predict_emotion(text)
        print(f"Detected Emotion: {detected_emotion}")

        # Step 2: Generate speech based on detected emotion
        output_file = generate_speech(text, detected_emotion)  # Removed speaker_wav

        if output_file:
            # Return the relative path to the generated speech file
            return jsonify({"emotion": detected_emotion, "audio_file": f"/{output_file}"})
        else:
            return jsonify({"error": "Failed to generate speech"}), 500
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
