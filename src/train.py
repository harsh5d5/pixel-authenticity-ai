import os
import cv2
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Import our custom modules
from preprocessing import ImagePreprocessor
from extractors import ForensicExtractors
from forgery_detectors import ForgeryDetectors

class ForensicTrainer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.preprocessor = ImagePreprocessor()
        self.extractors = ForensicExtractors()
        self.detectors = ForgeryDetectors()
        self.features_list = []

    def collect_features(self):
        """Iterates through data folders and extracts forensic features."""
        from tqdm import tqdm
        categories = {'training_real': 0, 'training_fake': 1}
        
        print(f"Starting feature extraction from: {self.data_dir}")
        
        for category, label in categories.items():
            dataset_path = os.path.join(self.data_dir, category)
            if not os.path.exists(dataset_path):
                print(f"Warning: Path {dataset_path} does not exist. Skipping.")
                continue
            
            # Recursively find all images
            all_files = []
            for root, dirs, files in os.walk(dataset_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                        all_files.append(os.path.join(root, f))
            
            print(f"Found {len(all_files)} images in '{category}'...")
            
            # Optionally limit to avoid extreme long runs if testing
            # all_files = all_files[:2500] 

            for img_path in tqdm(all_files, desc=f"Extracting {category}"):
                try:
                    filename = os.path.basename(img_path)
                    
                    # 1. Preprocess
                    processed_data = self.preprocessor.process(img_path)
                    
                    # 2. Extract Evidence Features
                    forensic_features = self.extractors.extract_all_features(processed_data)
                    
                    # 3. Extract Forgery Pattern Features
                    forgery_features = self.detectors.get_forgery_report(
                        processed_data['original_standardized'], 
                        processed_data['noise_map']
                    )
                    
                    # Merge all features
                    combined_features = {**forensic_features, **forgery_features}
                    combined_features['label'] = label
                    combined_features['filename'] = filename
                    
                    self.features_list.append(combined_features)
                    
                except Exception as e:
                    # Silence errors for large batches but log them to a file if needed
                    pass

        # Convert to DataFrame
        df = pd.DataFrame(self.features_list)
        if not df.empty:
            df.to_csv('forensic_features.csv', index=False)
            print("Feature extraction complete. Data saved to 'forensic_features.csv'.")
        return df

    def train_model(self, df):
        """Trains a Random Forest classifier on the extracted features."""
        if df.empty:
            print("No data found to train on. Please check your data directory.")
            return None

        # Prepare features (X) and labels (y)
        # We drop filename and label from X
        X = df.drop(['label', 'filename'], axis=1)
        y = df['label']

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

        # Initialize and train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print("\n--- Model Performance ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Save the model
        with open('forensic_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("\nTrained model saved as 'forensic_model.pkl'.")
        
        # Display Feature Importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop Evidence Indicators:")
        print(importances)

        return model

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    
    # Use the specific data path
    dataset_path = os.path.join(os.getcwd(), 'data')
    trainer = ForensicTrainer(data_dir=dataset_path)
    
    # 1. Collect features (this will create 'forensic_features.csv')
    df = trainer.collect_features()
    
    # 2. Train and save model
    if not df.empty:
        trainer.train_model(df)
    else:
        print(f"\n[!] Dataset at {dataset_path} is empty or not found.")

