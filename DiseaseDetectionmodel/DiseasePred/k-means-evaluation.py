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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define ANSI color codes
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

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
    
    print(f"\n{BLUE}Predicted Cluster: {cluster}{RESET}")
    print(f"{BLUE}Disease: {disease_prediction}{RESET}")
    print(f"{YELLOW}Description: {description_list.get(disease_prediction, 'No description available.')}{RESET}")
    print(f"{YELLOW}Precautions:{RESET}")
    for i, precaution in enumerate(precautionDictionary.get(disease_prediction, []), 1):
        print(f"{YELLOW}{i}) {precaution}{RESET}")

    doctor = doctors[doctors['disease'] == disease_prediction]
    if not doctor.empty:
        print(f"{YELLOW}Consult: {doctor['name'].values[0]}{RESET}")
        print(f"{YELLOW}Visit: {doctor['link'].values[0]}{RESET}")
    else:
        print(f"{YELLOW}No doctor details available.{RESET}")

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    symptoms = extract_symptoms(user_input)
    
    if symptoms:
        print(f"Extracted Symptoms: {symptoms}")
        predict_cluster_with_severity(symptoms, severityDictionary)
    else:
        print("Could not extract symptoms. Please try again.")


# Step 1: Map clusters to ground truth labels
mapped_labels = np.zeros_like(train_clusters)

for cluster in range(n_clusters):
    # Find indices of samples in this cluster
    mask = (train_clusters == cluster)
    # Get the most common ground truth label for this cluster
    most_common = Counter(y_train[mask]).most_common(1)[0][0]
    mapped_labels[mask] = most_common

# Step 2: Evaluate the metrics
# Compare mapped cluster labels with the ground truth
conf_matrix = confusion_matrix(y_train, mapped_labels)
accuracy = accuracy_score(y_train, mapped_labels)
precision = precision_score(y_train, mapped_labels, average='weighted')
recall = recall_score(y_train, mapped_labels, average='weighted')
f1 = f1_score(y_train, mapped_labels, average='weighted')

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)




cm = confusion_matrix(y_test, kmeans.predict(x_test))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

