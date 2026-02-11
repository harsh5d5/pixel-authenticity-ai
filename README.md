# Neural Trust Forensic Engine

An image forensics project to detect manipulated images using OpenCV and Machine Learning.

## Project Structure
- `data/`: Contains training/testing images.
    - `real/`: Original, un-manipulated images.
    - `fake/`: Manipulated or AI-generated images.
- `src/`: Source code for the project.
    - `preprocessing.py`: Image normalization and cleaning.
    - `extractors.py`: Feature extraction (ELA, FFT, Noise, etc.).
    - `train.py`: Model training and evaluation.
- `requirements.txt`: Project dependencies.
- `detect.py`: Main script for inference on single images.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your images in the `data/real` and `data/fake` folders.
3. Run the processing and training scripts.
