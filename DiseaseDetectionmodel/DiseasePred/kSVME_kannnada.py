import re
import sys
import pandas as pd
import gtts
import os
import pyttsx3
import speech_recognition as sr
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from playsound import playsound 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
# color = sys.stdout.shell
from colorama import init,Fore, Style

# Initialize colorama
init(autoreset=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)

clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train.values,y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
#print (scores.mean())

model=SVC()
model.fit(x_train.values,y_train)
#print("for svm: ")
#print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 150)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def speak_kannada(text):
    print(text)
    tts = gtts.gTTS(text, lang='kn')  # Set language to Kannada
    tts.save("kannada_audio.mp3")
    playsound("kannada_audio.mp3")
    os.remove("kannada_audio.mp3")
    
    
    
severityDictionary=dict()
description_list = dict()
description_list1 = dict()
precautionDictionary=dict()
precautionDictionary1=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
        sum=sum+severityDictionary[item]
        #import pdb;pdb.set_trace()
    if((sum*days)/(len(exp)+1)>13):
        # print(Fore.YELLOW +"You should take the consultation from doctor.","COMMENT")
        print(Fore.YELLOW + "ನೀವು ವೈದ್ಯರ ಸಲಹೆಯನ್ನು ಪಡೆಯುವುದು ಉತ್ತಮ.", "COMMENT")
        # readn("You should take the consultation from doctor.")
        speak_kannada("ನೀವು ವೈದ್ಯರ ಸಲಹೆಯನ್ನು ಪಡೆಯುವುದು ಉತ್ತಮ.")
        print()
    else:
        # print(Fore.YELLOW +"It might not be that bad but you should take precautions." ,"STRING")
        print(Fore.YELLOW + "ಅದು ಬಹಳ ಕಠಿಣವಲ್ಲದಿರಬಹುದು, ಆದರೆ ನೀವು ಮುನ್ನೆಚ್ಚರಿಕೆ ತೆಗೆದುಕೊಳ್ಳಬೇಕು.", "STRING")
        # readn("It might not be that bad, but you should take precautions.")
        speak_kannada("ಅದು ಬಹಳ ಕಠಿಣವಲ್ಲದಿರಬಹುದು, ಆದರೆ ನೀವು ಮುನ್ನೆಚ್ಚರಿಕೆ ತೆಗೆದುಕೊಳ್ಳಬೇಕು.")

        print()

def getDescription():
    global description_list
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getDescription1():
    global description_list1
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_desc_kannada.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list1.update(_description)

def getSeverityDict():
    global severityDictionary
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def getprecautionDict1():
    global precautionDictionary1
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_precaution_Kannada.csv', encoding='utf-8') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            # _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            # precautionDictionary1.update(_prec)
            # Ensure the row has at least 5 columns before accessing
            if len(row) >= 5:
                _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
                precautionDictionary1.update(_prec)
            else:
                print(f"Skipping incomplete row: {row}")

#readn("Hi, I am HealthCare Chatbot, If you like to interact with me through Voice Command please type 0, to continue our conversation through keyboard input type 1")
# readn("Hi")

# select=int(input())

def getInfo():
    speak_kannada("ನಮಸ್ಕಾರ")
    t1 = gtts.gTTS("ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಹೇಳಿ", lang='kn')
    t1.save("name.mp3")
    playsound("name.mp3")
    os.remove("name.mp3")
    #readn("ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಹೇಳಿ")
    ear = sr.Recognizer()
    with sr.Microphone() as inputs:
        name=ear.listen(inputs)
    print("ನಮಸ್ಕಾರ " + ear.recognize_google(name, language = 'kn-IN'))
    t2 = gtts.gTTS("ನಮಸ್ಕಾರ " + ear.recognize_google(name, language = 'kn'))
    t2.save("name1.mp3")
    playsound("name1.mp3")
    os.remove("name1.mp3")
    #readn("ನಮಸ್ಕಾರ" + ear.recognize_google(name, language = 'kn-IN'))
    
def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
      
def sec_predict(symptoms_exp):
    df = pd.read_csv(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\Data\Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train.values, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

# Load Kannada-to-English mapping
def load_kannada_to_english_mapping():
    mapping = {}
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\kannada_to_english.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            kannada_symptom = row[0].strip()
            english_symptom = row[1].strip()
            mapping[kannada_symptom] = english_symptom
    return mapping
  
# Initialize mapping
kannada_to_english_mapping = load_kannada_to_english_mapping()

def load_english_to_kannada_mapping():
    mapping = {}
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\kannada_to_english.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            kannada_symptom = row[0].strip()
            english_symptom = row[1].strip()
            mapping[english_symptom] = kannada_symptom
    return mapping
english_to_kannada_mapping = load_english_to_kannada_mapping()

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        t3 = gtts.gTTS("ದಯವಿಟ್ಟು ನೀವು ಅನುಭವಿಸುತ್ತಿರುವ ರೋಗಲಕ್ಷಣವನ್ನು ಹೇಳಿ",lang='kn')
        t3.save("name2.mp3")
        playsound("name2.mp3")
        os.remove("name2.mp3")
        #readn("ದಯವಿಟ್ಟು ನೀವು ಅನುಭವಿಸುತ್ತಿರುವ ರೋಗಲಕ್ಷಣವನ್ನು ಹೇಳಿ")
        ear = sr.Recognizer()
        with sr.Microphone() as inputs:
            disease_input=ear.listen(inputs)
             # Recognize speech input and print it
            recognized_text = ear.recognize_google(disease_input, language='kn')
            print(f"Recognized Input: {recognized_text}")  # Log the recognized input
            
             # Translate Kannada input to English
            translated_symptom = kannada_to_english_mapping.get(recognized_text.strip(), None)
            if not translated_symptom:
                print("The entered symptom is not recognized.")
                t7 = gtts.gTTS("ಕ್ಷಮಿಸಿ, ನೀವು ನಮೂದಿಸಿದ ರೋಗಲಕ್ಷಣವನ್ನು ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸಿ.", lang='kn')
                t7.save("name6.mp3")
                playsound("name6.mp3")
                os.remove("name6.mp3")
                continue
            
            print(f"Translated Symptom to English: {translated_symptom}")
            
            
              # Proceed with the recognized input
            conf,cnf_dis=check_pattern(chk_dis,translated_symptom)
            
            
        if conf==1:
            t4 = gtts.gTTS("ನೀವು ಹೇಳಿರುವ ಲಕ್ಷಣಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಹುಡುಕಾಟಗಳು",lang='kn')
            t4.save("name3.mp3")
            playsound("name3.mp3")
            os.remove("name3.mp3")
            #readn("ನೀವು ಹೇಳಿರುವ ಲಕ್ಷಣಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಹುಡುಕಾಟಗಳು")
            for num,it in enumerate(cnf_dis,21):
                # it=it.replace('_',' ')
                # print(num,")",it)
                # t5 = gtts.gTTS(f"{num,it}")
                # t5.save("name4.mp3")
                # playsound("name4.mp3")
                # os.remove("name4.mp3")
                # #readn(f"{num,it}")
                # it=it.replace(' ','_')
                kannada_symptom = english_to_kannada_mapping.get(it, it).replace('_', ' ')
                print(num, ")", kannada_symptom)
                # Speak the Kannada symptom
                t5 = gtts.gTTS(f"{num} {kannada_symptom}", lang='kn')
                t5.save("name4.mp3")
                playsound("name4.mp3")
                os.remove("name4.mp3")
            if num!=21:
                t6 = gtts.gTTS(f"ನೀವು ಉದ್ದೇಶಿಸಿರುವ ಒಂದನ್ನು ಆಯ್ಕೆಮಾಡಿ (21 - {num})",lang='en')
                t6.save("name5.mp3")
                playsound("name5.mp3")
                os.remove("name5.mp3")
                #readn(f"ನೀವು ಉದ್ದೇಶಿಸಿರುವ ಒಂದನ್ನು ಆಯ್ಕೆಮಾಡಿ (21 - {num})")
                ear = sr.Recognizer()
                with sr.Microphone() as inputs:
                    conf_inp=ear.listen(inputs)
                    conf_inp = int((ear.recognize_google(conf_inp, language = 'en-IN')))
            else:
                conf_inp=21
            disease_input=cnf_dis[conf_inp-21]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            t7 = gtts.gTTS("ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ರೋಗಲಕ್ಷಣವನ್ನು ನಮೂದಿಸಿ.",lang='kn')
            t7.save("name6.mp3")
            playsound("name6.mp3")
            os.remove("name6.mp3")
            #readn("ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ರೋಗಲಕ್ಷಣವನ್ನು ನಮೂದಿಸಿ.")
    while True:
        try:
            t8 = gtts.gTTS("ನೀವು ಈ ರೋಗಲಕ್ಷಣವನ್ನು ಎಷ್ಟು ದಿನಗಳಿಂದ ಅನುಭವಿಸುತ್ತಿದ್ದೀರಿ?",lang='kn')
            t8.save("name7.mp3")
            playsound("name7.mp3")
            os.remove("name7.mp3")
            #readn("ನೀವು ಈ ರೋಗಲಕ್ಷಣವನ್ನು ಎಷ್ಟು ದಿನಗಳಿಂದ ಅನುಭವಿಸುತ್ತಿದ್ದೀರಿ?")
            ear = sr.Recognizer()
            with sr.Microphone() as inputs:
                num_days=ear.listen(inputs)
                num_days = int((ear.recognize_google(num_days, language = 'kn-IN')))
            print(num_days)
            num_days = 7 if num_days >20 else num_days
            #import pdb;pdb.set_trace()
            break
        except:
            print(num_days)
            t9 = gtts.gTTS("ದಯವಿಟ್ಟು ಸರಿಯಾದ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಿ.",lang='kn')
            t9.save("name8.mp3")
            playsound("name8.mp3")
            os.remove("name8.mp3")
            #readn("ದಯವಿಟ್ಟು ಸರಿಯಾದ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳಿ.")
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
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            #readn(f"Are you experiencing any of the following symptoms, type yes to say?")
            t10 = gtts.gTTS("ನೀವು ಈ ಕೆಳಗಿನ ಯಾವುದೇ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಅನುಭವಿಸುತ್ತಿದ್ದೀರಾ?, ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.",lang='kn')
            t10.save("name9.mp3")
            playsound("name9.mp3")
            os.remove("name9.mp3")
            #readn(f"ನೀವು ಈ ಕೆಳಗಿನ ಯಾವುದೇ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಅನುಭವಿಸುತ್ತಿದ್ದೀರಾ?, ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                # syms=syms.replace('_',' ')
                # print(syms)
                # t11 = gtts.gTTS(f"{syms}")
                # t11.save("name10.mp3")
                # playsound("name10.mp3")
                # os.remove("name10.mp3")
                # Translate symptom to Kannada
                kannada_symptom = english_to_kannada_mapping.get(syms, syms).replace('_', ' ')
                print(kannada_symptom)

                t11 = gtts.gTTS(f"{kannada_symptom}", lang='kn')
                t11.save("name10.mp3")
                playsound("name10.mp3")
                os.remove("name10.mp3")
                
                #readn(f"{syms}")
                while True:
                    ear = sr.Recognizer()
                    with sr.Microphone() as inputs:
                        inp=ear.listen(inputs)
                        inp = str((ear.recognize_google(inp, language = 'kn')))
                    if(inp=="ಹೌದು" or inp=="ಇಲ್ಲ"):
                        break
                    else:
                        t12 = gtts.gTTS("ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳುವ ಮೂಲಕ ಸರಿಯಾದ ಉತ್ತರಗಳನ್ನು ಒದಗಿಸಿ",lang='kn')
                        t12.save("name11.mp3")
                        playsound("name11.mp3")
                        os.remove("name11.mp3")
                if(inp=="ಹೌದು"):
                    syms=syms.replace(' ','_')
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು", present_disease[0])
                t13 = gtts.gTTS(f"ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು,{present_disease[0]}",lang='kn')
                t13.save("name12.mp3")
                playsound("name12.mp3")
                os.remove("name12.mp3")
                #readn(f"ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು {present_disease[0]}")
                print()
                t14 = gtts.gTTS(f"{description_list1[present_disease[0]]}",lang='kn')
                t14.save("name13.mp3")
                playsound("name13.mp3")
                os.remove("name13.mp3")
                #readn(f"{description_list[present_disease[0]]}")
                print(description_list1[present_disease[0]])
                row = doctors[doctors['disease'] == present_disease[0]]
                print()
                print('ಸಮಾಲೋಚಿಸಿ',str(row['name'].values))
                t15 = gtts.gTTS(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}")
                t15.save("name14.mp3")
                playsound("name14.mp3")
                os.remove("name14.mp3")
                #readn(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}")
                print()
                t16 = gtts.gTTS("ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಲಿಂಕ್‌ಗೆ ಭೇಟಿ ನೀಡಿ",lang='kn')
                t16.save("name15.mp3")
                playsound("name15.mp3")
                os.remove("name15.mp3")
                #readn("ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಲಿಂಕ್‌ಗೆ ಭೇಟಿ ನೀಡಿ")
                print(str(row['link'].values))
                print()
                #readn(f"You may have {present_disease[0]}")
                #readn(f"{description_list[present_disease[0]]}")

            else:
                print("ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು", present_disease[0])
                t17 = gtts.gTTS(f"ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು,{present_disease[0]} or {second_prediction[0]}")
                t17.save("name16.mp3")
                playsound("name16.mp3")
                os.remove("name16.mp3")
                #readn(f"ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು{present_disease[0]} or {second_prediction[0]}")
                print()
                print(description_list1[present_disease[0]])
                t18 = gtts.gTTS(f"{description_list1[present_disease[0]]}",lang='kn')
                t18.save("name17.mp3")
                playsound("name17.mp3")
                os.remove("name17.mp3")
                #readn(f"{description_list[present_disease[0]]}")
                print(description_list1[second_prediction[0]])
                t19 = gtts.gTTS(f"{description_list1[second_prediction[0]]}",lang='kn')
                t19.save("name18.mp3")
                playsound("name18.mp3")
                os.remove("name18.mp3")
                #readn(f"{description_list[second_prediction[0]]}")
                row = doctors[doctors['disease'] == present_disease[0]]
                print()
                print('ಸಮಾಲೋಚಿಸಿ',str(row['name'].values))
                t20 = gtts.gTTS(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}",lang='kn')
                t20.save("name19.mp3")
                playsound("name19.mp3")
                os.remove("name19.mp3")
                #readn(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}")
                print()
                t21 = gtts.gTTS("ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಲಿಂಕ್‌ಗೆ ಭೇಟಿ ನೀಡಿ",lang='kn')
                t21.save("name20.mp3")
                playsound("name20.mp3")
                os.remove("name20.mp3")
                #readn("ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಲು ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಲಿಂಕ್‌ಗೆ ಭೇಟಿ ನೀಡಿ")
                print(f"{row['link'].values}")
                print()

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary1[present_disease[0]]
            t22 = gtts.gTTS("ಕೆಳಗಿನ ಕ್ರಮಗಳನ್ನು ಕೈಗೊಳ್ಳಿ",lang='kn')
            t22.save("name21.mp3")
            playsound("name21.mp3")
            os.remove("name21.mp3")
            #readn("ಕೆಳಗಿನ ಕ್ರಮಗಳನ್ನು ಕೈಗೊಳ್ಳಿ")
            print()
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
                speak_kannada(f"{i+1, j}")

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)
doc_dataset = pd.read_csv(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\doctors_dataset.csv', names = ['Name', 'Description'])
diseases = reduced_data.index
diseases = pd.DataFrame(diseases)
doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan
doctors['disease'] = diseases['prognosis']
doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']
record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']
# getSeverityDict()
# getDescription()
# getDescription1()
# getprecautionDict()
# getprecautionDict1()
# getInfo()
# tree_to_code(clf,cols)
#import pdb;pdb.set_trace()
# Existing code...

if __name__ == "__main__":
    getInfo()  # This will no longer run on import
    getSeverityDict()
    getDescription()
    getDescription1()
    getprecautionDict()
    getprecautionDict1()
    tree_to_code(clf,cols)

# print("----------------------------------------------------------------------------------------")