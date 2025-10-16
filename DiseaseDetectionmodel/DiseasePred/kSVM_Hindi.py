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

training = pd.read_csv(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\Data\Training.csv')
testing= pd.read_csv(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\Data\Testing.csv')
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

def speak_hindi(text):
    tts = gtts.gTTS(text, lang='hi')  # Set language to Kannada
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
        print(Fore.YELLOW + "आपके लिए डॉक्टर की सलाह लेना बेहतर होगा।", "COMMENT")
        # readn("You should take the consultation from doctor.")
        speak_hindi("आपको डॉक्टर की सलाह लेना बेहतर है।")
        print()
    else:
        # print(Fore.YELLOW +"It might not be that bad but you should take precautions." ,"STRING")
        print(Fore.YELLOW + "यह बहुत कठिन नहीं हो सकता, लेकिन आपको सावधानी बरतनी चाहिए।", "STRING")
        # readn("It might not be that bad, but you should take precautions.")
        speak_hindi("यह बहुत कठिन नहीं हो सकता, लेकिन आपको सावधानी बरतनी चाहिए।")

        print()

def hindigetDescription():
    global description_list
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def hindigetDescription1():
    global description_list1
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_desc_hindi.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list1.update(_description)

def hindigetSeverityDict():
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

def hindigetprecautionDict():
    global precautionDictionary
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def hindigetprecautionDict1():
    global precautionDictionary1
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\symptom_precaution_hindi.csv', encoding='utf-8') as csv_file:

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

def hindigetInfo():
    speak_hindi("नमस्कार")
    t1 = gtts.gTTS("कृपया अपना नाम बताएं", lang='hi')
    t1.save("name.mp3")
    playsound("name.mp3")
    os.remove("name.mp3")
    ear = sr.Recognizer()
    with sr.Microphone() as inputs:
        name=ear.listen(inputs)
    print("नमस्कार " + ear.recognize_google(name, language = 'hi-IN'))
    t2 = gtts.gTTS("नमस्कार " + ear.recognize_google(name, language = 'hi'))
    t2.save("name1.mp3")
    playsound("name1.mp3")
    os.remove("name1.mp3")
    
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
    df = pd.read_csv('Data/Training.csv')
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
def load_hindi_to_english_mapping():
    mapping = {}
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\hindi_to_english.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            hindi_symptom = row[0].strip()
            english_symptom = row[1].strip()
            mapping[hindi_symptom] = english_symptom
    return mapping
  
# Initialize mapping
hindi_to_english_mapping = load_hindi_to_english_mapping()

def load_english_to_hindi_mapping():
    mapping = {}
    with open(r'C:\Users\adthi\Desktop\Mini\Disease Detection model- OG\MasterData\hindi_to_english.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            hindi_symptom = row[0].strip()
            english_symptom = row[1].strip()
            mapping[english_symptom] = hindi_symptom
    return mapping
english_to_hindi_mapping = load_english_to_hindi_mapping()

def hinditree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        t3 = gtts.gTTS("कृपया आप जो लक्षण महसूस कर रहे हैं, वह बताएं", lang='hi')
        t3.save("name2.mp3")
        playsound("name2.mp3")
        os.remove("name2.mp3")
        ear = sr.Recognizer()
        with sr.Microphone() as inputs:
            disease_input=ear.listen(inputs)
             # Recognize speech input and print it
            recognized_text = ear.recognize_google(disease_input, language='hi')
            print(f"Recognized Input: {recognized_text}")  # Log the recognized input
            
             # Translate Kannada input to English
            translated_symptom = hindi_to_english_mapping.get(recognized_text.strip(), None)
            if not translated_symptom:
                print("The entered symptom is not recognized.")
                t7 = gtts.gTTS("क्षमा करें, आप द्वारा दिए गए लक्षणों को पहचानना संभव नहीं है। कृपया फिर से प्रयास करें।", lang='hi')
                t7.save("name6.mp3")
                playsound("name6.mp3")
                os.remove("name6.mp3")
                continue
            
            print(f"Translated Symptom to English: {translated_symptom}")
            
            
              # Proceed with the recognized input
            conf,cnf_dis=check_pattern(chk_dis,translated_symptom)
            
            
        if conf==1:
            t4 = gtts.gTTS("आप द्वारा बताए गए लक्षण से संबंधित खोजें", lang='hi')
            t4.save("name3.mp3")
            playsound("name3.mp3")
            os.remove("name3.mp3")
            for num,it in enumerate(cnf_dis,21):
                hindi_symptom = english_to_hindi_mapping.get(it, it).replace('_', ' ')
                print(num, ")", hindi_symptom)
                # Speak the Kannada symptom
                t5 = gtts.gTTS(f"{num} {hindi_symptom}", lang='hi')
                t5.save("name4.mp3")
                playsound("name4.mp3")
                os.remove("name4.mp3")
            if num!=21:
                t6 = gtts.gTTS(f"कृपया वह विकल्प चुनें जिसे आप चाहते हैं (21 - {num})", lang='hi')
                t6.save("name5.mp3")
                playsound("name5.mp3")
                os.remove("name5.mp3")
                ear = sr.Recognizer()
                with sr.Microphone() as inputs:
                    conf_inp=ear.listen(inputs)
                    conf_inp = int((ear.recognize_google(conf_inp, language = 'en-IN')))
            else:
                conf_inp=21
            disease_input=cnf_dis[conf_inp-21]
            break
            
        else:
            t7 = gtts.gTTS("कृपया एक मान्य लक्षण दर्ज करें।", lang='hi')
            t7.save("name6.mp3")
            playsound("name6.mp3")
            os.remove("name6.mp3")
            #readn("ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ರೋಗಲಕ್ಷಣವನ್ನು ನಮೂದಿಸಿ.")
    while True:
        try:
            t8 = gtts.gTTS("आप यह लक्षण कितने दिनों से महसूस कर रहे हैं?", lang='hi')
            t8.save("name7.mp3")
            playsound("name7.mp3")
            os.remove("name7.mp3")
            ear = sr.Recognizer()
            with sr.Microphone() as inputs:
                num_days=ear.listen(inputs)
                num_days = int((ear.recognize_google(num_days, language = 'hi-IN')))
            print(num_days)
            num_days = 7 if num_days >20 else num_days
            break
        except:
            print(num_days)
            t9 = gtts.gTTS("कृपया सही संख्या बताएं।", lang='hi')
            t9.save("name8.mp3")
            playsound("name8.mp3")
            os.remove("name8.mp3")
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
            t10 = gtts.gTTS("क्या आप निम्नलिखित में से कोई लक्षण महसूस कर रहे हैं? कृपया हां या नहीं कहें।", lang='hi')
            t10.save("name9.mp3")
            playsound("name9.mp3")
            os.remove("name9.mp3")
            #readn(f"ನೀವು ಈ ಕೆಳಗಿನ ಯಾವುದೇ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಅನುಭವಿಸುತ್ತಿದ್ದೀರಾ?, ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                hindi_symptom = english_to_hindi_mapping.get(syms, syms).replace('_', ' ')
                print(hindi_symptom)

                t11 = gtts.gTTS(f"{hindi_symptom}", lang='hi')
                t11.save("name10.mp3")
                playsound("name10.mp3")
                os.remove("name10.mp3")
                
                #readn(f"{syms}")
                while True:
                    ear = sr.Recognizer()
                    with sr.Microphone() as inputs:
                        inp=ear.listen(inputs)
                        inp = str((ear.recognize_google(inp, language = 'hi')))
                    if(inp=="हाँ" or inp=="नहीं"):
                        break
                    else:
                        t12 = gtts.gTTS("कृपया हां या नहीं कहकर सही उत्तर प्रदान करें।", lang='hi')
                        t12.save("name11.mp3")
                        playsound("name11.mp3")
                        os.remove("name11.mp3")
                if(inp!="नहीं"):
                    syms=syms.replace(' ','_')
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("आपको यह रोग हो सकता है", present_disease[0])
                t13 = gtts.gTTS(f"आपको यह रोग हो सकता है, {present_disease[0]}", lang='hi')
                t13.save("name12.mp3")
                playsound("name12.mp3")
                os.remove("name12.mp3")
                #readn(f"ನೀವು ಈ ರೋಗವನ್ನು ಹೊಂದಿರಬಹುದು {present_disease[0]}")
                print()
                t14 = gtts.gTTS(f"{description_list1[present_disease[0]]}",lang='hi')
                t14.save("name13.mp3")
                playsound("name13.mp3")
                os.remove("name13.mp3")
                #readn(f"{description_list[present_disease[0]]}")
                print(description_list1[present_disease[0]])
                row = doctors[doctors['disease'] == present_disease[0]]
                print()
                print('परामर्श करें',str(row['name'].values))
                t15 = gtts.gTTS(f"{'परामर्श करें',str(row['name'].values)}",lang='hi')
                t15.save("name14.mp3")
                playsound("name14.mp3")
                os.remove("name14.mp3")
                #readn(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}")
                print()
                t16 = gtts.gTTS("ऑनलाइन डॉक्टर से संपर्क करने के लिए कृपया नीचे दिए गए लिंक पर जाएं।", lang='hi')
                t16.save("name15.mp3")
                playsound("name15.mp3")
                os.remove("name15.mp3")
                print(str(row['link'].values))
                print()
               
            else:
                print("आपको यह रोग हो सकता है", present_disease[0])
                t17 = gtts.gTTS(f"आपको यह रोग हो सकता है, {present_disease[0]} या {second_prediction[0]}", lang='hi')
                t17.save("name16.mp3")
                playsound("name16.mp3")
                os.remove("name16.mp3")
                print()
                print(description_list1[present_disease[0]])
                t18 = gtts.gTTS(f"{description_list1[present_disease[0]]}",lang='hi')
                t18.save("name17.mp3")
                playsound("name17.mp3")
                os.remove("name17.mp3")
                print(description_list1[second_prediction[0]])
                
                t19 = gtts.gTTS(f"{description_list1[second_prediction[0]]}",lang='hi')
                t19.save("name18.mp3")
                playsound("name18.mp3")
                os.remove("name18.mp3")
                
                row = doctors[doctors['disease'] == present_disease[0]]
                print()
                print('ಸಮಾಲೋಚಿಸಿ',str(row['name'].values))
                t20 = gtts.gTTS(f"{'ಸಮಾಲೋಚಿಸಿ',str(row['name'].values)}")
                t20.save("name19.mp3")
                playsound("name19.mp3")
                os.remove("name19.mp3")

                print()
                t21 = gtts.gTTS("कृपया ऑनलाइन डॉक्टर से संपर्क करने के लिए नीचे दिए गए लिंक पर जाएं।", lang='hi')
                t21.save("name20.mp3")
                playsound("name20.mp3")
                os.remove("name20.mp3")

                print(f"{row['link'].values}")
                print()

            precution_list=precautionDictionary1[present_disease[0]]
            t22 = gtts.gTTS("नीचे दिए गए कदमों को उठाएं।", lang='hi')
            t22.save("name21.mp3")
            playsound("name21.mp3")
            os.remove("name21.mp3")
            print()
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
                speak_hindi(f"{i+1, j}")


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

if __name__ == "__main__":
    hindigetSeverityDict()
    hindigetDescription()
    hindigetDescription1()
    hindigetprecautionDict()
    hindigetprecautionDict1()
    hindigetInfo()
    hinditree_to_code(clf,cols)
#import pdb;pdb.set_trace()
# print("----------------------------------------------------------------------------------------")