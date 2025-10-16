import cv2
import re
import sys
import pandas as pd
import pyttsx3
import speech_recognition as sr
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import csv
import warnings
from colorama import init, Fore, Style
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Initialize colorama
init(autoreset=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load class names and model configuration
classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Global dictionaries
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

# Load data functions
def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 2:
                _description = {row[0]: row[1]}
                description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 2:  # Add this condition to ensure rows have at least 2 elements
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 5:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precautionDictionary.update(_prec)

def get_svm_result():
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    y1 = y

    reduced_data = training.groupby(training['prognosis']).max()

    # Mapping strings to numbers
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)

    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train.values, y_train)

    scores = cross_val_score(clf, x_test, y_test, cv=3)
    print("===========================================================")
    print(scores)
    print(scores.mean())

    model = SVC()
    model.fit(x_train.values, y_train)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    def readn(nstr):
        engine = pyttsx3.init()
        engine.setProperty('voice', "english+f5")
        engine.setProperty('rate', 130)
        engine.say(nstr)
        engine.runAndWait()
        engine.stop()

    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I did not get that.")
                return None

    def calc_condition(exp, days):
        sum = 0
        for item in exp:
            sum = sum + severityDictionary[item]
        if (sum * days) / (len(exp) + 1) > 13:
            print(Fore.YELLOW + "You should take the consultation from doctor.")
            readn("You should take the consultation from doctor.")
            print()
        else:
            print(Fore.GREEN + "It might not be that bad but you should take precautions.", "STRING")
            readn("It might not be that bad but you should take precautions.")
            print()

    def getInfo():
        print("----------------HealthCare ChatBot-----------------------------------")
        readn("Your Name?")
        name = recognize_speech()
        if not name:
            name = input("Your Name? \t\t\t\t->")
        print("Hello, ", name)
        readn(f"Hello, {name}")

    def check_pattern(dis_list, inp):
        pred_list = []
        inp = inp.replace(' ', '_')
        patt = f"{inp}"
        regexp = re.compile(patt)
        pred_list = [item for item in dis_list if regexp.search(item)]
        if len(pred_list) > 0:
            return 1, pred_list
        else:
            return 0, []

    def sec_predict(symptoms_exp):
        df = pd.read_csv('Data/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train.values, y_train)

        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[symptoms_dict[item]] = 1

        return rf_clf.predict([input_vector])

    def print_disease(node):
        node = node[0]
        val = node.nonzero()
        disease = le.inverse_transform(val[0])
        return list(map(lambda x: x.strip(), list(disease)))

    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")
        symptoms_present = []

        while True:
            readn("Enter the symptom you are experiencing")
            disease_input = recognize_speech()
            if not disease_input:
                disease_input = input("\nEnter the symptom you are experiencing  \t\t->")
            conf, cnf_dis = check_pattern(chk_dis, disease_input)
            if conf == 1:
                print("Searches related to input: ")
                for num, it in enumerate(cnf_dis):
                    print(num, ")", it)
                readn(f"Select the one you meant (0 - {num}): ")
                conf_inp = recognize_speech()
                if not conf_inp:
                    conf_inp = input(f"Select the one you meant (0 - {num}):  ")
                else:
                    conf_inp = int(conf_inp)

                disease_input = cnf_dis[int(conf_inp)]

                break
            else:
                readn("Enter valid symptom.")
                print("Enter valid symptom.")

        while True:
            readn("Okay. From how many days?")
            num_days = recognize_speech()
            if not num_days:
                num_days = int(input("Okay. From how many days? : "))

            else:
                num_days = int(num_days)
            if num_days:
                num_days = 7 if num_days > 20 else num_days
                break
            else:
                print("Enter valid input.")

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                red_cols = reduced_data.columns 
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                print(Fore.BLUE +"Are you experiencing any ")
                readn("Are you experiencing any of the following symptoms?")
                symptoms_exp = []
                for syms in list(symptoms_given):
                    inp = ""
                    print(syms, "? : ", end='')
                    readn(syms + "?")
                    while True:
                        inp = recognize_speech()
                        if not inp:
                            inp = input()
                        if inp.lower() == "yes" or inp.lower() == "no":
                            break
                        else:
                            print("Provide proper answers i.e. yes/ no.")
                            readn("Provide proper answers i.e. yes/ no.")
                    if inp.lower() == "yes":
                        symptoms_exp.append(syms)
                    if inp.lower() == "no":
                        continue
                symptoms_exp.extend(symptoms_present)
                print(symptoms_exp)
                readn("Your likely Disease is ")
                readn("Disease :")
                readn(present_disease)
                readn(f"was predicted with {scores[prediction] * 100:.2f}% accuracy.")
                readn("The symptoms you have are")
                readn(symptoms_exp)
                print("The symptoms you have are : ")
                print(symptoms_exp)
                calc_condition(symptoms_exp, num_days)
                print(Fore.GREEN + "The model has suggest the consult a doctor based on the symptoms. you seem to be have been a doctor who treats related to the predicted disease")


    getInfo()
    getDescription()
    getSeverityDict()
    getprecautionDict()

    root = tk.Tk()
    root.title("SVM for Detection of Disease")
    root.geometry("600x500")

    style = ttk.Style()
    style.theme_use('clam')

    # Initialize threading and speech recognition
    threading.Thread(target=recognize_and_analyze).start()
    root.mainloop()

