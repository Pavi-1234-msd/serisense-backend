import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app, origins=['*'])

# ── Model Loading ─────────────────────────────────────────
model = None
CLASS_NAMES = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']

def load_model():
    global model
    
    # Try .keras format first
    keras_path = os.path.join(os.path.dirname(__file__), 'model', 'mulberry_model.keras')
    h5_path = os.path.join(os.path.dirname(__file__), 'model', 'mulberry_model.h5')
    
    print(f"Current directory: {os.path.dirname(__file__)}")
    print(f"Model dir contents: {os.listdir(os.path.join(os.path.dirname(__file__), 'model'))}")
    
    # Try keras format
    if os.path.exists(keras_path):
        print(f"Found .keras file: {keras_path}")
        print(f"File size: {os.path.getsize(keras_path) / (1024*1024):.1f} MB")
        try:
            model = tf.keras.models.load_model(keras_path, compile=False)
            print(f"✅ Model loaded from .keras!")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return
        except Exception as e:
            print(f"❌ .keras load failed: {e}")
    
    # Try h5 format
    if os.path.exists(h5_path):
        print(f"Found .h5 file: {h5_path}")
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
            print(f"✅ Model loaded from .h5!")
            return
        except Exception as e:
            print(f"❌ .h5 load failed: {e}")
    
    print("❌ No model file found or all load attempts failed")

# Load at startup
load_model()