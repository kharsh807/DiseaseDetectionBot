import threading
import streamlit as st
import os
from k_means_NEMO import extract_symptoms, predict_cluster_with_severity, getSeverityDict, severityDictionary
from k_means_Riva import speech_extract_symptoms, speech_predict_cluster_with_severity, precautionDictionary, description_list
from rag_report import process_pdf
from ultralytics import YOLO
import cv2
import serial
import numpy as np
import time
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
from kSVME_kannnada import clf, cols, tree_to_code, getInfo, getSeverityDict, getDescription, getDescription1, getprecautionDict, getprecautionDict1
from kSVM_Hindi import clf, cols, hinditree_to_code, hindigetInfo, hindigetSeverityDict, hindigetDescription, hindigetDescription1, hindigetprecautionDict, hindigetprecautionDict1
tts_engine = pyttsx3.init()

def speak(text):
    """Text-to-speech function in a separate thread."""
    print(text)
    threading.Thread(target=_speak_in_background, args=(text,)).start()

def _speak_in_background(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def load_severity_data():
    getSeverityDict()

def init_arduino():
    try:
        arduino = serial.Serial('COM5', 9600)  # Replace 'COM9' with your Arduino port
        time.sleep(2)  # Wait for the connection to establish
        return arduino
    except serial.SerialException as e:
        st.error(f"Error initializing Arduino: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error initializing Arduino: {str(e)}")
        return None

def run_object_detection(arduino):
    st.write("Starting Object Detection...")
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    arduino = init_arduino()
    if arduino:
        arduino.write(b'M')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        person_detected = False
        for det in results[0].boxes:
            if det.cls == 0:
                person_detected = True
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if arduino:
            if person_detected:
                arduino.write(b'P')
                speak("Please move away")
                time.sleep(2)
            else:
                arduino.write(b'O')

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

st.title("Disease Prediction & Medical Analysis App")
st.sidebar.header("Navigation")
option = st.sidebar.radio("Go to:", ["Home", "Text", "Speech", "Kannada Model","Hindi Model", "Report Summary"])

if "start_detection" not in st.session_state:
    st.session_state.start_detection = False

if option == "Home":
    st.write("Welcome to the Disease Prediction & Medical Analysis App!")
    st.write("Choose a feature from the sidebar to begin:")
    st.write("1. **Text**: Input symptoms in text form for prediction.")
    st.write("2. **Speech**: Speak your symptoms and get predictions.")
    st.write("3. **Kannada Model**: Interact with the app in Kannada.")
    st.write("4. **Hindi Model**: Interact with the app in Hindi.")
    st.write("5. **Report Summary**: Upload a medical report to get a detailed summary.")
   

elif option == "Text":
    st.header("Disease Prediction (Text)")
    user_input = st.text_input("Enter your symptoms:")

    if st.button("Predict Disease"):
        if user_input:
            load_severity_data()
            symptoms = extract_symptoms(user_input)

            if symptoms:
                st.write(f"Extracted Symptoms: {symptoms}")
                import io
                import sys

                output = io.StringIO()
                sys.stdout = output
                predict_cluster_with_severity(symptoms, severityDictionary)
                sys.stdout = sys.__stdout__

                st.text(output.getvalue())
                st.session_state.navigation_ready = True
            else:
                st.error("Could not extract symptoms. Please try again.")
        else:
            st.error("Please enter your symptoms.")

elif option == "Speech":
    st.header("Disease Prediction (Speech)")
    st.write("Speak your symptoms into the microphone.")

    if st.button("Start Listening"):
        import io
        import sys
        ear = sr.Recognizer()

        try:
            st.write("Listening...")
            recognized_text = ""

            with sr.Microphone() as inputs:
                audio = ear.listen(inputs)
                recognized_text = ear.recognize_google(audio)

            if recognized_text:
                st.write(f"Recognized Text: {recognized_text}")
                load_severity_data()
                symptoms = speech_extract_symptoms(recognized_text)

                if symptoms:
                    st.write(f"Extracted Symptoms: {symptoms}")
                    output = io.StringIO()
                    sys.stdout = output
                    disease, precautions = speech_predict_cluster_with_severity(symptoms, severityDictionary)
                    sys.stdout = sys.__stdout__

                    st.text(output.getvalue())
                    st.session_state.navigation_ready = True
                else:
                    st.error("Could not extract symptoms. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")



elif option == "Kannada Model":
    st.header("Disease Prediction (Kannada Model)")
    st.write("Interact with the app in Kannada.")

    if st.button("Start Kannada Interaction"):
        import io
        import sys

        
        try:
            # output = io.StringIO()
            # sys.stdout = output
            getInfo()  # Collect user information in Kannada
            getSeverityDict()
            getDescription()
            getDescription1()
            getprecautionDict()
            getprecautionDict1()
            tree_to_code(clf,cols)  # Run the decision tree-based Kannada model
            # sys.stdout = sys.__stdout__
            # st.text(output.getvalue())
        except Exception as e:
            st.error(f"Error in Kannada model: {str(e)}")


elif option == "Hindi Model":
    st.header("Disease Prediction (Hindi Model)")
    st.write("Interact with the app in Hindi.")

    if st.button("Start Hindi Interaction"):
        import io
        import sys

        
        try:
            # output = io.StringIO()
            # sys.stdout = output
            hindigetInfo()  # Collect user information in Kannada
            hindigetSeverityDict()
            hindigetDescription()
            hindigetDescription1()
            hindigetprecautionDict()
            hindigetprecautionDict1()
            hinditree_to_code(clf,cols)  # Run the decision tree-based Kannada model
            # sys.stdout = sys.__stdout__
            # st.text(output.getvalue())
        except Exception as e:
            st.error(f"Error in Hindi model: {str(e)}")
            

elif option == "Report Summary":
    st.header("Medical Report Summary")
    uploaded_file = st.file_uploader("Upload your medical report (PDF):", type=["pdf"])

    if uploaded_file:
        if st.button("Provide Summary"):
            try:
                query = "Summarize the patient's diagnosis and treatment history in a detailed manner."
                summary = process_pdf(uploaded_file, query)
                st.subheader("Report Summary:")
                st.write(summary)

            except Exception as e:
                st.error(f"Error processing report: {str(e)}")
                
                
            
if st.session_state.get("navigation_ready", False):
    if st.button("Click here to navigate to the department"):
        st.session_state.start_detection = True
        st.session_state.navigation_ready = False

if st.session_state.start_detection:
    st.write("Navigating to the department...")
    arduino = init_arduino()
    if arduino:
        run_object_detection(arduino)
    st.session_state.start_detection = False
