import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load features
df = pd.read_csv('forensic_features.csv')
X = df.drop(['label'], axis=1)
y = df['label']

# Split same as training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
with open('forensic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
