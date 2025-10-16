import csv
import re
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
from openai import OpenAI
from ultralytics import YOLO
import cv2

# Define ANSI color codes
#= '\033[94m'
#= '\033[93m'
#= '\033[0m'

# Initialize and suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing datasets
training = pd.read_csv('Data/Training.csv')
cols = training.columns[:-1]  # All columns except the prognosis column
x = training[cols]
y = training['prognosis']

# Label encoding for prognosis
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Splitting the data (if needed for validation)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

# K-means clustering model
n_clusters = len(y.unique())
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(x_train.values)

# Assign each sample to a cluster
train_clusters = kmeans.predict(x_train.values)

# Initialize severity dictionary
severityDictionary = {}

# Load severity data from CSV file
def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) < 2:  # Skip rows that don't have at least two columns
                continue
            try:
                symptom = row[0].strip().lower()
                severity = int(row[1].strip())
                severityDictionary[symptom] = severity
            except ValueError:
                print(f"Warning: Skipping row due to format issue: {row}")

getSeverityDict()

# Load additional data files for descriptions, precautions, and doctor details
description_list = {}
precautionDictionary = {}
doctors = pd.read_csv('MasterData/doctors_dataset.csv', names=['name', 'link'])
diseases = pd.DataFrame(training['prognosis'].unique(), columns=['disease'])
doctors['disease'] = diseases['disease']

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = row[1:]

getDescription()
getprecautionDict()

# Map clusters to diseases
cluster_to_disease = {}
for cluster in range(n_clusters):
    cluster_indices = (train_clusters == cluster)
    common_disease = pd.Series(y_train[cluster_indices]).mode()[0]
    disease_name = le.inverse_transform([common_disease])[0]
    cluster_to_disease[cluster] = disease_name

# Initialize the NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Function to extract symptoms from user input
def extract_symptoms(user_input):
    """
    Function to extract symptoms from user input using NVIDIA LLM API.
    """
    try:
        # Sending the user input to the model for symptom extraction
        completion = client.chat.completions.create(
            model="nv-mistralai/mistral-nemo-12b-instruct",
            messages=[{"role": "user", "content": f"Extract symptoms from this sentence. For multi-word symptoms, use underscores and lowercase: {user_input}"}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        
        # Collect and process the response chunks
        extracted_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                extracted_text += chunk.choices[0].delta.content

        # Clean and normalize the extracted symptoms
        symptoms = [
            symptom.strip().lstrip('-').strip().replace('\n', '')
            for symptom in re.split(r'[\n,-]', extracted_text)  # Split by newline, comma, or dash
        ]

        # Filter out any empty strings and return as a list
        return [symptom for symptom in symptoms if symptom]
    
    except Exception as e:
        print(f"Error extracting symptoms: {e}")
        return []


# Function to predict cluster and map to disease
def predict_cluster_with_severity(symptom_list, severity_dict):
    symptom_vector = np.zeros(len(cols))
    for symptom in symptom_list:
        if symptom in cols:
            severity = severity_dict.get(symptom, 1)  # Default severity 1 if not found
            symptom_vector[cols.get_loc(symptom)] = severity

    cluster = kmeans.predict([symptom_vector])[0]
    disease_prediction = cluster_to_disease.get(cluster, "Unknown Disease")
    
    print(f"\nPredicted Cluster: {cluster}")
    print(f"Disease: {disease_prediction}")
    print(f"Description: {description_list.get(disease_prediction, 'No description available.')}")
    print(f"Precautions:")
    for i, precaution in enumerate(precautionDictionary.get(disease_prediction, []), 1):
        print(f"{i}) {precaution}")

    doctor = doctors[doctors['disease'] == disease_prediction]
    if not doctor.empty:
        print(f"Consult: {doctor['name'].values[0]}")
        print(f"Visit: {doctor['link'].values[0]}")
    else:
        print(f"No doctor details available.")

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    symptoms = extract_symptoms(user_input)
    
    if symptoms:
        print(f"Extracted Symptoms: {symptoms}")
        predict_cluster_with_severity(symptoms, severityDictionary)
    else:
        print("Could not extract symptoms. Please try again.")
