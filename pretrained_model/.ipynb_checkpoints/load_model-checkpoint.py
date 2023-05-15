from urllib.request import urlopen  # Importing the 'urlopen' function from the 'urllib.request' module to open URLs
import joblib  # Importing the 'joblib' module for saving and loading Python objects
import pandas as pd  # Importing the 'pandas' library for data manipulation and analysis
import numpy as np  # Importing the 'numpy' library for numerical operations
from sklearn.feature_extraction.text import CountVectorizer  # Importing the 'CountVectorizer' class for text feature extraction
from sklearn.preprocessing import MultiLabelBinarizer  # Importing the 'MultiLabelBinarizer' class for transforming multilabel data
from sklearn.model_selection import train_test_split  # Importing the 'train_test_split' function for splitting data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # Importing the 'RandomForestClassifier' class for random forest classification
from sklearn.metrics import classification_report  # Importing the 'classification_report' function for generating classification performance metrics

model = joblib.load("./Model_compressed.joblib")  # Loading a pre-trained machine learning model from a joblib file
vectorizer = joblib.load("./Vectorizer_compressed.joblib")  # Loading a pre-trained CountVectorizer from a joblib file
mlb = joblib.load("./MLB_compressed.joblib")  # Loading a pre-trained MultiLabelBinarizer from a joblib file

def predict_attributes(verb):
    verb_vectorized = vectorizer.transform([verb])  # Transforming the input verb into a vector representation
    predicted_attributes = model.predict(verb_vectorized)  # Predicting the attributes for the input verb
    return mlb.inverse_transform(predicted_attributes)[0]  # Inverse transforming the predicted attributes back to their original form


while True:
    input_verb = [str(input("Enter a verb: "))]  # Taking user input for a verb
    print(predict_attributes(str(input_verb)))  # Predicting and printing the attributes for the input verb
