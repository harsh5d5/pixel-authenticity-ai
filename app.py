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
CORS(app) # Allow frontend to talk to backend

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
            prediction = model.predict(X)[0]
            
            # Trust score should represent the probability of being REAL
            trust_score = probability[0] * 100
            
            # Only label as FAKE if the probability of REAL is very low (bias correction)
            # This makes the "Neural Scan" more conservative before crying wolf
            if trust_score < 40:
                prediction_label = "FAKE"
            else:
                prediction_label = "REAL"
        
        # 4. Generate Heatmap for Frontend
        ela_map = extractors.run_ela(data['original_standardized'])
        _, buffer = cv2.imencode('.jpg', ela_map)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 5. Build Human-Readable Evidence
        evidence_points = []
        if forensic_features['ela_mean'] > 5.0:
            evidence_points.append({"text": "Compression anomalies detected in pixel blocks.", "status": "alert"})
        else:
            evidence_points.append({"text": "Compression levels are uniform across the image.", "status": "valid"})
            
        if combined_features['noise_inconsistency'] > 50:
            evidence_points.append({"text": "Inconsistent noise patterns suggest splicing.", "status": "alert"})
        else:
            evidence_points.append({"text": "Sensor noise matches authentic hardware profile.", "status": "valid"})

        if forensic_features['fft_mean'] > 180:
            evidence_points.append({"text": "Post-processing or AI-upscaling traces found.", "status": "alert"})
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
    app.run(port=5000, debug=True)
