from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for, session, flash
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import uuid
from werkzeug.utils import secure_filename
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from werkzeug.security import generate_password_hash, check_password_hash

# Import DB module
import db
from datetime import datetime

# Optional: TF-IDF retrieval over dataset text
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-key')

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_disease_model.h5')
CLASS_FILE = os.path.join(MODEL_DIR, 'class_indices.json')

# Load class labels saved during training if available
if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, 'r', encoding='utf-8') as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ['acne', 'rash', 'eczema', 'allergy', 'fungal']

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

MODEL = load_model()

# Load suggestions mapping if present
SUGGESTIONS_FILE = os.path.join(MODEL_DIR, 'suggestions.json')
if os.path.exists(SUGGESTIONS_FILE):
    try:
        with open(SUGGESTIONS_FILE, 'r', encoding='utf-8') as f:
            SUGGESTIONS = json.load(f)
    except Exception:
        SUGGESTIONS = {}
else:
    SUGGESTIONS = {}

def prepare_image(image, target=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    arr = np.array(image)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Load precomputed dataset embeddings for fast similarity search (optional)
EMB_PATH = os.path.join(MODEL_DIR, 'embeddings.npy')
EMB_META = os.path.join(MODEL_DIR, 'embeddings_meta.json')
DATA_EMB = None
DATA_META = None
if os.path.exists(EMB_PATH) and os.path.exists(EMB_META):
    try:
        DATA_EMB = np.load(EMB_PATH)
        with open(EMB_META, 'r', encoding='utf-8') as f:
            DATA_META = json.load(f)
        norms = np.linalg.norm(DATA_EMB, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        DATA_EMB = DATA_EMB / norms
    except Exception:
        DATA_EMB = None
        DATA_META = None

# Embedding model for uploaded images
EMB_MODEL = None
if DATA_EMB is not None:
    try:
        EMB_MODEL = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    except Exception:
        EMB_MODEL = None

# Initialize database
db.init_db()

@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)

def login_required(fn):
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Now using Neon PostgreSQL
        user_record = db.get_user(username)
        
        if user_record and check_password_hash(user_record['password'], password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        
        flash('Invalid credentials', 'danger')
        return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
            
        existing_user = db.get_user(username)
        if existing_user:
            flash('Username already taken', 'danger')
            return render_template('register.html')
            
        hashed_password = generate_password_hash(password)
        try:
            db.create_user(username, hashed_password)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'danger')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = session.get('user')
    return render_template('dashboard.html', user=user, classes=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not found. Train the model and place skin_disease_model.h5 in /model.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    file = request.files['file']
    user = session.get('user')
    try:
        # Save the file uniquely to display in history later
        ext = 'jpg'
        if '.' in file.filename:
            ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(BASE_DIR, 'static', 'uploads', filename)
        
        file.seek(0)
        file.save(filepath)
        
        # open the saved file for prediction
        img = Image.open(filepath)
        x = prepare_image(img)
        preds = MODEL.predict(x)[0]
        # build top-k list for frontend (sorted highest probabilities)
        top_k = 3
        idxs = np.argsort(preds)[::-1][:top_k]
        top_list = []
        for i in idxs:
            lab = CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else str(i)
            score = float(preds[i])
            top_list.append({'class': lab, 'confidence': score})
        # primary prediction is the highest
        idx = int(idxs[0])
        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
        confidence = float(preds[idx])

        result = {'class': label, 'confidence': confidence, 'top': top_list}

        # If embeddings available, compute nearest neighbor
        matched_path = None
        matched_label = None
        if DATA_EMB is not None and EMB_MODEL is not None:
            try:
                emb_x = EMB_MODEL.predict(x)
                emb_x = emb_x / (np.linalg.norm(emb_x, axis=1, keepdims=True) + 1e-10)
                sims = np.dot(DATA_EMB, emb_x[0])
                best = int(np.argmax(sims))
                meta = DATA_META[best]
                
                orig_path = meta['path'].replace('\\', '/')
                if '/model/' in orig_path:
                    rel = 'model/' + orig_path.split('/model/')[1]
                else:
                    rel = os.path.basename(orig_path)
                
                matched_path = rel
                matched_label = meta.get('label', '')
                result.update({'matched_path': rel, 'matched_label': matched_label, 'similarity': float(sims[best])})
            except Exception as e:
                print(f"Similarity search error: {e}")
                pass

        # Save to history
        db.save_scan(user, label, confidence, top_list, filename, matched_path, matched_label)

        # attach suggestions for the predicted label (if available)
        try:
            if label in SUGGESTIONS:
                result['suggestions'] = SUGGESTIONS[label]
            else:
                # fallback: include a generic disclaimer
                result['suggestions'] = {'disclaimer': 'This is informational only. Consult a healthcare professional for diagnosis and treatment.'}
        except Exception:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_ai_insight', methods=['POST'])
@login_required
def generate_ai_insight():
    data = request.get_json(force=True, silent=True) or {}
    label = data.get('label', '')
    
    if not label:
        return jsonify({'error': 'Label is required'}), 400
        
    hf_api_key = os.environ.get('HF_API_KEY')
    if not hf_api_key:
        return jsonify({'error': 'Hugging Face API key not configured.'}), 500
        
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are Dr. Derma AI, a highly skilled but concise dermatologist assistant AI. Provide informative insights.\nCRITICAL: Include a strong medical disclaimer that you are an AI and the user MUST consult a real doctor for serious issues. Formulate your response as safe advice. Format your response cleanly using Markdown lists and bold text. Keep it brief and focused."},
            {"role": "user", "content": f"The patient's scan was identified as: {label}.\nPlease provide exactly 3 things:\n1. Is this condition typically serious? Under what symptoms should they visit a doctor immediately?\n2. What are safe home remedies to soothe this?\n3. What are standard over-the-counter medicines, soaps, or creams that might be helpful?"}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_data = response.json()
        if 'choices' in res_data and len(res_data['choices']) > 0:
            generated_text = res_data['choices'][0]['message']['content']
            return jsonify({'insight': generated_text.strip()})
        else:
            return jsonify({'error': 'Invalid response from AI server.'}), 500
    except requests.exceptions.RequestException as e:
        print(f"HF API Error: {e}")
        return jsonify({'error': 'Failed to connect to AI server. Please try again later.'}), 503

@app.route('/history')
@login_required
def history():
    user = session.get('user')
    rows = db.get_history(user)
    # Convert result to list of dicts
    history_list = []
    for r in rows:
        history_list.append({
            'id': r['id'],
            'label': r['label'],
            'confidence': float(r['confidence']),
            'top_k': json.loads(r['top_k']) if r['top_k'] else [],
            'image_path': r['image_path'],
            'matched_path': r.get('matched_path'),
            'matched_label': r.get('matched_label'),
            'timestamp': r['timestamp'].isoformat() if isinstance(r['timestamp'], datetime) else r['timestamp']
        })
    return jsonify(history_list)

@app.route('/delete-scan/<int:scan_id>', methods=['DELETE'])
@login_required
def delete_scan(scan_id):
    user = session.get('user')
    try:
        db.delete_scan(scan_id, user)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting scan {scan_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/matched')
def matched():
    # Serve matched dataset image by relative path under `model/`.
    path = request.args.get('path')
    if not path:
        abort(404)
    # normalize and ensure inside BASE_DIR
    safe = os.path.normpath(os.path.join(BASE_DIR, path))
    if not safe.startswith(os.path.normpath(BASE_DIR)):
        abort(403)
    if not os.path.exists(safe):
        abort(404)
    return send_file(safe)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Expects JSON: {"question": "...", "label": "optional-predicted-label"}
    data = request.get_json(force=True, silent=True) or {}
    question = data.get('question', '').strip()
    label = data.get('label', 'an unspecified condition')
    
    if not question:
        return jsonify({'error': 'Question is required.'}), 400
        
    hf_api_key = os.environ.get('HF_API_KEY')
    if not hf_api_key:
        return jsonify({'error': 'Hugging Face API key not configured.'}), 500
        
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are Dr. Derma AI, a highly skilled but concise dermatologist assistant AI. Provide informative, contextual health guidance based on the patient's questions.\nCRITICAL: Include a strong medical disclaimer that you are an AI and the user MUST consult a real doctor for serious issues. Format your response cleanly using Markdown lists and bold text."},
            {"role": "user", "content": f"Context: The patient recently scanned an image that was identified as: {label}.\nPatient Question: {question}"}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_data = response.json()
        if 'choices' in res_data and len(res_data['choices']) > 0:
            generated_text = res_data['choices'][0]['message']['content']
            return jsonify({'question': question, 'answer': generated_text.strip()})
        else:
            return jsonify({'error': 'Invalid response from AI server.'}), 500
    except requests.exceptions.RequestException as e:
        print(f"HF API Chat Error: {e}")
        return jsonify({'error': 'Failed to connect to AI server. Please try again later.'}), 503

if __name__ == '__main__':
    # Run without the reloader to avoid automatic restarts when auxiliary files change
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(host=host, port=port, debug=debug_mode, use_reloader=True)
