import os
import sys
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing import ImagePreprocessor
from extractors import ForensicExtractors
from forgery_detectors import ForgeryDetectors

app = Flask(__name__)
# Enable full CORS for local development
CORS(app, resources={r"/*": {"origins": "*"}})

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Load the trained model
MODEL_PATH = 'forensic_model.pkl'
if not os.path.exists(MODEL_PATH):
    print("WARNING: forensic_model.pkl not found. Please run src/train.py first.")
    model = None
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

preprocessor = ImagePreprocessor()
extractors = ForensicExtractors()
detectors = ForgeryDetectors()

@app.route('/')
def home():
    return jsonify({
        "status": "Neural Trust Backend is Running",
        "api_endpoint": "/analyze",
        "method": "POST"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img_path = "temp_upload.jpg"
    file.save(img_path)
    
    try:
        # 1. Process Image
        data = preprocessor.process(img_path)
        
        # 2. Extract Features
        forensic_features = extractors.extract_all_features(data)
        forgery_features = detectors.get_forgery_report(
            data['original_standardized'], 
            data['noise_map']
        )
        
        combined_features = {**forensic_features, **forgery_features}
        
        # 3. Predict using the trained model
        feature_order = [
            'ela_mean', 'ela_std', 'fft_mean', 'texture_variance', 
            'noise_mean', 'copy_move_score', 'noise_inconsistency'
        ]
        X = [combined_features[f] for f in feature_order]
        X = np.array(X).reshape(1, -1)
        
        prediction_label = "REAL"
        trust_score = 50.0 
        
        if model:
            probability = model.predict_proba(X)[0] # [prob_real, prob_fake]
            
            # Trust score represents the probability of being REAL
            trust_score = probability[0] * 100
            
            if trust_score < 35:
                prediction_label = "FAKE"
            elif 35 <= trust_score < 70:
                prediction_label = "EDITED"
            else:
                prediction_label = "REAL"
        
        # 4. Generate Heatmap for Frontend
        ela_map = extractors.run_ela(data['original_standardized'])
        _, buffer = cv2.imencode('.jpg', ela_map)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 5. Build Human-Readable Evidence
        evidence_points = []
        
        # ELA Evidence
        if forensic_features['ela_mean'] > 8.0:
            evidence_points.append({"text": "High compression anomalies detected (Significant editing).", "status": "alert"})
        elif forensic_features['ela_mean'] > 4.0:
            evidence_points.append({"text": "Minor compression inconsistencies metadata.", "status": "warning"})
        else:
            evidence_points.append({"text": "Compression levels are uniform across the image.", "status": "valid"})
            
        # Noise Evidence
        if combined_features['noise_inconsistency'] > 70:
            evidence_points.append({"text": "Major noise inconsistency (Likely splicing).", "status": "alert"})
        elif combined_features['noise_inconsistency'] > 35:
            evidence_points.append({"text": "Slight variance in sensor noise profile.", "status": "warning"})
        else:
            evidence_points.append({"text": "Sensor noise matches authentic hardware profile.", "status": "valid"})

        # FFT Evidence
        if forensic_features['fft_mean'] > 200:
            evidence_points.append({"text": "Artificial frequency spikes (AI patterns).", "status": "alert"})
        elif forensic_features['fft_mean'] > 150:
            evidence_points.append({"text": "Unusual frequency signatures detected.", "status": "warning"})
        else:
            evidence_points.append({"text": "No unnatural frequency spikes detected.", "status": "valid"})

        return jsonify({
            "status": prediction_label,
            "trust_score": round(trust_score, 2),
            "evidence": evidence_points,
            "heatmap": f"data:image/jpeg;base64,{heatmap_base64}",
            "metrics": {
                "ela": min(100, (1 - (forensic_features['ela_mean'] / 15)) * 100),
                "noise": min(100, (1 - (combined_features['noise_inconsistency'] / 100)) * 100),
                "fft": min(100, (1 - (forensic_features['fft_mean'] / 255)) * 100)
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
