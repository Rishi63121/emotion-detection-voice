import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import extract_features

def train_model(data_path='data/'):
    print("Starting model training...")
    features = []
    labels = []

    # Loop through each audio file
    for file in os.listdir(data_path):
        if file.endswith(".wav"):
            print(f"Processing file: {file}")
            emotion = file.split("-")[2]  # Example: '03-01-05-01-02-01-12.wav' → '05'
            file_path = os.path.join(data_path, file)

            feat = extract_features(file_path)
            features.append(feat)
            labels.append(emotion)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Train the model
    print("Training the Random Forest model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print("Saving the trained model...")
    joblib.dump(model, "emotion_model.pkl")
    print("Model saved as emotion_model.pkl")

    print("Calculating accuracy...")
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    train_model()  # Start training
