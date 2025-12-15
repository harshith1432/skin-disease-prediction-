from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for, session, flash
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

# Optional: TF-IDF retrieval over dataset text
try:
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
	SKLEARN_AVAILABLE = True
except Exception:
	SKLEARN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-key')

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

# Simple demo credential (for production, use a proper user store)
DEMO_USERS = {'admin': 'password'}

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


def answer_question(question: str, label: str = None):
	"""Simple label-aware answer generator using SUGGESTIONS.
	This is a rule-based helper — not an LLM. It returns a short answer string or dict.
	"""
	q = (question or '').lower()
	info = None
	if label and label in SUGGESTIONS:
		info = SUGGESTIONS[label]

	# prioritize direct intents
	if 'home' in q or 'remedy' in q or 'home remedy' in q:
		if info and info.get('home_remedies'):
			return {'type': 'home_remedies', 'items': info['home_remedies']}
		return {'type': 'generic', 'answer': 'General home care: keep skin clean, use gentle moisturizers, avoid irritants. See a doctor if it worsens.'}

	if 'medicine' in q or 'drug' in q or 'treatment' in q or 'prescribe' in q:
		if info and info.get('medicines'):
			return {'type': 'medicines', 'items': info['medicines']}
		return {'type': 'generic', 'answer': 'Medicines depend on diagnosis; OTC options are listed in the suggestions panel. Consult a physician for prescriptions.'}

	if 'symptom' in q or 'how to identify' in q or 'how do i know' in q or 'what is' in q:
		if info:
			out = []
			if info.get('short'): out.append(info['short'])
			if info.get('do'): out.append('Key care: ' + '; '.join(info.get('do', [])[:3]))
			return {'type': 'summary', 'answer': ' '.join(out)}
		return {'type': 'generic', 'answer': 'Symptoms vary; inspect for rash, scaling, inflammation, and follow guidance in the suggestions panel.'}

	if 'when' in q and ('doctor' in q or 'see' in q or 'urgent' in q):
		return {'type': 'escalation', 'answer': 'Seek medical attention if you have spreading rash, severe pain, fever, difficulty breathing, or swelling of the face.'}

	# fallback: if we have class info, give an overview
	if info:
		summary = info.get('short') or info.get('title')
		return {'type': 'summary', 'answer': summary}

	return {'type': 'generic', 'answer': 'I can provide general care suggestions and home remedies. Ask about medicines, home care, or when to see a doctor.'}


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

# Retrieval index (TF-IDF) built from DATA_META (label + filename + folder)
RETR_VECTORIZER = None
RETR_MATRIX = None
RETR_DOCS = None

def build_retrieval_index():
	global RETR_VECTORIZER, RETR_MATRIX, RETR_DOCS
	if not SKLEARN_AVAILABLE:
		return
	if not DATA_META:
		return
	docs = []
	for m in DATA_META:
		path = m.get('path','')
		label = m.get('label','')
		name = os.path.splitext(os.path.basename(path))[0]
		parent = os.path.basename(os.path.dirname(path))
		text = ' '.join([label, name.replace('_',' '), parent.replace('_',' ')])
		docs.append(text)
	if not docs:
		return
	try:
		vec = TfidfVectorizer(stop_words='english', max_features=20000)
		mat = vec.fit_transform(docs)
		RETR_VECTORIZER = vec
		RETR_MATRIX = mat
		RETR_DOCS = DATA_META.copy()
	except Exception:
		RETR_VECTORIZER = None
		RETR_MATRIX = None
		RETR_DOCS = None

try:
	build_retrieval_index()
except Exception:
	pass


# Embedding model for uploaded images
EMB_MODEL = None
if DATA_EMB is not None:
	try:
		EMB_MODEL = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
	except Exception:
		EMB_MODEL = None


@app.route('/')
def index():
	if session.get('user'):
		return redirect(url_for('dashboard'))
	return redirect(url_for('login'))


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
		if username in DEMO_USERS and DEMO_USERS[username] == password:
			session['user'] = username
			return redirect(url_for('dashboard'))
		flash('Invalid credentials', 'danger')
		return render_template('login.html')
	return render_template('login.html')


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
	try:
		img = Image.open(file.stream)
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
		if DATA_EMB is not None and EMB_MODEL is not None:
			try:
				emb_x = EMB_MODEL.predict(x)
				emb_x = emb_x / (np.linalg.norm(emb_x, axis=1, keepdims=True) + 1e-10)
				sims = np.dot(DATA_EMB, emb_x[0])
				best = int(np.argmax(sims))
				meta = DATA_META[best]
				rel = os.path.relpath(meta['path'], BASE_DIR)
				result.update({'matched_path': rel.replace('\\','/'), 'matched_label': meta.get('label', ''), 'similarity': float(sims[best])})
			except Exception:
				# ignore embedding errors and continue
				pass

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
	label = data.get('label')
	if not question:
		return jsonify({'error': 'Question is required.'}), 400
	try:
		answer = answer_question(question, label)

		# Augment with retrieval results when available
		retrieved = []
		try:
			if SKLEARN_AVAILABLE and RETR_VECTORIZER is not None and RETR_MATRIX is not None and RETR_DOCS is not None:
				qtext = question
				if label:
					qtext = f"{label} " + qtext
				qv = RETR_VECTORIZER.transform([qtext])
				sims = cosine_similarity(qv, RETR_MATRIX)[0]
				if sims is not None and len(sims) > 0:
					top_k = min(5, len(sims))
					idxs = sims.argsort()[::-1][:top_k]
					for i in idxs:
						score = float(sims[i])
						if score <= 0:
							continue
						doc = RETR_DOCS[i]
						rel = os.path.relpath(doc.get('path',''), BASE_DIR).replace('\\','/')
						retrieved.append({'path': rel, 'label': doc.get('label',''), 'score': score})
		except Exception:
			retrieved = []

		out = {'question': question, 'answer': answer}
		if retrieved:
			out['retrieved'] = retrieved
		return jsonify(out)
	except Exception as e:
		return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	# Run without the reloader to avoid automatic restarts when auxiliary files change
	app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


