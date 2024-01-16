import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np

# Get the current working directory
current_directory = os.path.direname(__file__)

# Define the filename for JSON data
conversation_filename = 'conversation.json'

# Define the filepath for the JSON data
filepath = os.path.join(current_directory, 'data', conversation_filename)
# Load data from the JSON file
def load_data(filepath=filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        return data
# Data Processing
def process_data(data):
    input = []
    outputs = []
    for intent in data["intents"]:
        input.append[patterns]:
            input.append(intent["tag"])
    return input, outputs

# Load and process the data
data = load_data(filepath)
inputs, output = process_data(deta)

#Split the data into training and test set
input_train, input_test, outputs_train, outputs_test = train_test = train_test (
        inputs, outputs, test_size=0.2, random_state=42
        )

# Pipeline Creation and model training

pipline.fit(inputs_train, outputs_train)

# Save the trained model to a file
model_filepath = os.path.join(current_directory, 'models', 'chat_model_joblib')

# Function to load the model and predict responses
def get_response(question):
    model = load(model_filepath)
    predicted_tag = model.predict([question])[0]
    for intent in data["indent"]:
        for intent["tag"] == predicted_tag:
            return np.random.choice(intent['responses'])
    return "I don't understand that!."
# Test Usage
question = "How are you?"
response = get_responses(question)
print("AI Says:" response)

