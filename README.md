# 🛡️ NeuralTrust: Pixel Authenticity & Forensic Engine

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-93%25-brightgreen?style=for-the-badge)

**NeuralTrust** is a professional-grade image forensics platform designed to detect digital manipulations, AI-generated synthesis, and metadata inconsistencies. By merging traditional computer vision techniques with advanced Machine Learning, it delivers a comprehensive "Authenticity Report" for any digital asset.

---

## 🛠️ System Workflow

To maintain high transparency, our engine follows a multi-phase forensic pipeline. The flowchart below illustrates how an image moves from raw upload to a final security verdict.

![System Flowchart](image/flowchart.png)

*<b>Figure 1:</b> The technical architecture, showing the transition from Pre-processing to Feature Extraction and ML Inference.*

--- 

## 🚀 Key Forensic Capabilities

*   **🔍 Error Level Analysis (ELA):** Detects intentional alterations by highlighting inconsistencies in JPEG compression levels.
*   **📡 Frequency Domain Analysis (FFT):** Uncovers geometric artifacts and periodic patterns left behind by AI generators (GANs/Diffusion models).
*   **❄️ Noise Variance Analysis:** Scans for "sensor fingerprints" to identify spliced regions that don't match the original hardware noise profile.
*   **🤖 ML-Powered Verdict:** Executes a Random Forest classification model trained on over 10,000 deepfake and authentic samples.
*   **📊 Command Center:** A high-end, responsive dashboard utilizing glassmorphism for real-time visualization of forensic maps.

---

## 📂 Project Organization

The repository is structured to separate the core forensic engine from the user interface and API layers.

![Folder Structure](image/folder%20structure.png)

*<b>Figure 2:</b> Overview of the project directory, highlighting the modularity of the forensic extractors and the backend bridge.*

### Module Breakdown:
- **`client/`**: The frontend command center (HTML5/CSS3/JS).
- **`src/`**: The "Engine Room" containing preprocessing, feature extractors, and training logic.
- **`app.py`**: The RESTful API that powers the communication between the UI and the AI.
- **`detect.py`**: A low-latency CLI tool for rapid batch analysis.

---

## 💻 Setup & Installation

### 1. Environment Preparation
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-repo/neural-trust.git

# Install core dependencies
pip install -r requirements.txt
```

### 2. Launch the Platform
Start the backend server to enable real-time analysis:

```bash
python app.py
```
Once the server is running, simply open `client/index.html` in your web browser to start scanning images.

### 3. Model Training (Advanced)
If you wish to fine-tune the model for specific datasets:
1. Populate `data/training_real` and `data/training_fake`.
2. Execute the optimized training pipeline:
```bash
python src/train.py
```

---

## 🧪 Technologies
| Layer | Tech Stack |
| :--- | :--- |
| **Backend** | Python, Flask, Pickle |
| **Vision** | OpenCV, NumPy, Scikit-Image |
| **AI/ML** | Scikit-Learn (Random Forest) |
| **Frontend** | Vanilla JavaScript, CSS (Modern UI/UX) |

---

## ⚖️ License & Ethics
This tool is intended for research and educational purposes in the field of digital forensics. Redistributed under the **MIT License**.

---
*Created with ❤️ for a safer digital world.*
