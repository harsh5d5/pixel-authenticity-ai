import sys
import os
import cv2

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from preprocessing import ImagePreprocessor
from extractors import ForensicExtractors
from forgery_detectors import ForgeryDetectors

def run_quick_analysis(img_name):
    print(f"Analyzing {img_name}...")
    try:
        p = ImagePreprocessor()
        e = ForensicExtractors()
        d = ForgeryDetectors()
        
        data = p.process(img_name)
        features = e.extract_all_features(data)
        forgery = d.get_forgery_report(data['original_standardized'], data['noise_map'])
        
        print('\n--- FORENSIC ANALYSIS ---')
        print(f"ELA Score (Compression Inconsistency): {features['ela_mean']:.2f}")
        print(f"FFT Artifacts (Electronic/AI Signature): {features['fft_mean']:.2f}")
        print(f"Copy-Move Score (Cloning Detection): {forgery['copy_move_score']:.2f}")
        print(f"Noise Inconsistency (Splicing Detection): {forgery['noise_inconsistency']:.2f}")
        
        # Save heatmap
        ela_map = e.run_ela(data['original_standardized'])
        cv2.imwrite('forensic_heatmap.jpg', ela_map)
        print('\nHeatmap saved as: forensic_heatmap.jpg')
        
        # Simple Logic Interpretation
        if features['ela_mean'] > 5.0 or forgery['copy_move_score'] > 2.0:
            print("\nRESULT: 🚩 POTENTIAL MANIPULATION DETECTED")
            print("Reason: High compression error or detected texture cloning.")
        else:
            print("\nRESULT: ✅ LIKELY REAL / MINIMAL EDITING")
            print("Reason: Forensic signatures are consistent across the image.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    img = 'tigerwood4.jpg'
    run_quick_analysis(img)
