import os
import sys
import cv2
import pickle
import numpy as np
import argparse

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing import ImagePreprocessor
from extractors import ForensicExtractors
from forgery_detectors import ForgeryDetectors

class NeuralTrustDetector:
    def __init__(self, model_path='forensic_model.pkl'):
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.extractors = ForensicExtractors()
        self.detectors = ForgeryDetectors()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please run 'python src/train.py' first.")
            
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def analyze_image(self, image_path):
        """Processes a single image and returns the authenticity score and evidence."""
        # 1. Pipeline
        processed_data = self.preprocessor.process(image_path)
        
        # 2. Extract Features
        forensic_features = self.extractors.extract_all_features(processed_data)
        forgery_features = self.detectors.get_forgery_report(
            processed_data['original_standardized'], 
            processed_data['noise_map']
        )
        
        # Merge features for the model
        combined_features = {**forensic_features, **forgery_features}
        
        # Convert to list for model prediction (ensuring same order as training)
        # The order must match the columns in the training DataFrame
        feature_order = [
            'ela_mean', 'ela_std', 'fft_mean', 'texture_variance', 
            'noise_mean', 'copy_move_score', 'noise_inconsistency'
        ]
        X = [combined_features[f] for f in feature_order]
        X = np.array(X).reshape(1, -1)

        # 3. Predict
        probability = self.model.predict_proba(X)[0] # [Prob_Real, Prob_Fake]
        prediction = self.model.predict(X)[0]
        
        trust_score = probability[0] * 100 # Authentic percentage
        
        return {
            "prediction": "REAL" if prediction == 0 else "FAKE",
            "trust_score": f"{trust_score:.2f}%",
            "evidence": combined_features,
            "processed_image": processed_data['original_standardized']
        }

def main():
    parser = argparse.ArgumentParser(description='Neural Trust Forensic Detector')
    parser.add_argument('--image', type=str, help='Path to the image file to analyze')
    args = parser.parse_args()

    if not args.image:
        print("Usage: python detect.py --image path/to/image.jpg")
        return

    try:
        detector = NeuralTrustDetector()
        result = detector.analyze_image(args.image)

        print("\n" + "="*30)
        print("   NEURAL TRUST REPORT")
        print("="*30)
        print(f"FILE: {args.image}")
        print(f"RESULT: {result['prediction']}")
        print(f"AUTHENTICITY SCORE: {result['trust_score']}")
        print("-"*30)
        print("DETAILED EVIDENCE:")
        for key, value in result['evidence'].items():
            print(f" - {key}: {value:.4f}")
        print("="*30 + "\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
