import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np

# Get the current working directory
current_directory = os.path.dirname(__file__)

# Define the filepath for the JSON data
filepath = os.path.join(current_directory, '../data/conversation.json')

# Define the directory and filename for the model
model_directory = os.path.join(current_directory, 'models')
model_filename = 'chat_model_joblib'
model_filepath = os.path.join(current_directory, '../models/chat_model_joblib')
# Load data from the JSON file


def load_data(filepath=filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    intents = data["intents"]
    inputs = []
    outputs = []

    for intent in intents:
        for pattern in intent["patterns"]:
            inputs.append(pattern)
            outputs.append(intent["responses"])

    return inputs, outputs


# Split the data into training and test sets
inputs, outputs = load_data()
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# Create a TfidfVectorizer to convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the input training data using the vectorizer
inputs_train_tfidf = tfidf_vectorizer.fit_transform(inputs_train)

# Convert outputs_train to a 1D array or list
outputs_train = [label for labels in outputs_train for label in labels]

# Check if the number of samples matches
if inputs_train_tfidf.shape[0] != len(outputs_train):
    print("Number of input samples and output samples don't match. Adjusting...")

    # Truncate or pad outputs_train to match the number of input samples
    if inputs_train_tfidf.shape[0] < len(outputs_train):
        outputs_train = outputs_train[:inputs_train_tfidf.shape[0]]
    else:
        num_samples_to_add = inputs_train_tfidf.shape[0] - len(outputs_train)
        outputs_train.extend([''] * num_samples_to_add)

# Create a pipeline for the text classification model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Fit the model with the transformed input training data and updated outputs_train
pipeline.fit(inputs_train, outputs_train)

# Save the trained model to a file
dump(pipeline, model_filepath)

# Number of input samples
num_input_samples = len(inputs)

# Number of output samples
num_output_samples = len(outputs)

# Number of training samples
num_training_samples = len(inputs_train)

# Number of testing samples
num_testing_samples = len(inputs_test)

# Print the numbers
print("Number of input samples:", num_input_samples)
print("Number of output samples:", num_output_samples)
print("Number of training samples:", num_training_samples)
print("Number of testing samples:", num_testing_samples)
