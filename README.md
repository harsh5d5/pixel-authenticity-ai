# 🛡️ NeuralTrust: Pixel Authenticity & Forensic Engine

**NeuralTrust** is a state-of-the-art image forensics platform designed to detect digital manipulations, AI-generated synthesis, and metadata inconsistencies. By combining traditional forensic techniques with Machine Learning, it provides a "Trust Score" for any uploaded image.

---

## 🚀 Key Features

*   **🔍 Error Level Analysis (ELA):** Identifies differences in compression levels to spot spliced or edited regions.
*   **📡 Frequency Domain Analysis (FFT):** Detects unnatural checkerboard artifacts typical of AI generators (GANs/Diffusion).
*   **❄️ Noise Variance Analysis:** Scans for sensor noise inconsistencies that reveal image composition.
*   **🤖 ML-Powered Verdict:** Uses a Random Forest classifier (trained on 10,000+ samples) to provide a final Authenticity Score.
*   **📊 Dynamic Dashboard:** A premium, dark-themed command center for real-time analysis.

---

## 📂 Project Structure

```text
├── client/                # Modern Frontend Dashboard (HTML/CSS/JS)
├── src/                   # Core Forensic Engine
│   ├── preprocessing.py   # Image normalization & standardization
│   ├── extractors.py      # ELA, FFT, and Texture feature extraction
│   ├── forgery_detectors.py# Copy-Move & Noise inconsistency modules
│   └── train.py           # Model training pipeline
├── app.py                 # Flask REST API Backend
├── detect.py              # Command-line inference tool
├── forensic_model.pkl    # Pre-trained core model (93% Accuracy)
└── requirements.txt       # Project dependencies
```

---

## 🛠️ Setup & Installation

### 1. Clone & Install
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Engine
To start the forensic analysis platform:

```bash
# Start the Backend API
python app.py
```
Then, open `client/index.html` in your browser to access the dashboard.

### 3. Training (Optional)
To retrain the model with your own data:
1. Place images in `data/training_real` and `data/training_fake`.
2. Run the training script:
```bash
python src/train.py
```

---

## 🧪 Technologies Used
- **Backend:** Flask, Python
- **Computer Vision:** OpenCV, Scikit-Image, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Frontend:** Vanilla CSS (Glassmorphism), Javascript

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
