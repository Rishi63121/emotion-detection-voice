# predict.py
import pickle
from utils import extract_features  # Assuming you have this in 'utils.py'

# Load the trained model
with open('emotion_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Print confirmation that model is loaded
print("Model loaded successfully!")
