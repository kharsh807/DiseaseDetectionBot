# Disease Detection Bot

A comprehensive system for disease detection and medical assistance using various machine learning models and multilingual support.

## Project Structure

```
DiseaseDetectionBot/
├── DiseaseDetectionmodel/
│   ├── DiseasePred/
│   │   ├── Data/            # Training and testing datasets
│   │   ├── MasterData/      # Medical reference data
│   │   ├── models/          # ML models and related scripts
│   │   └── scripts/         # Python implementation scripts
├── ResultScreenShots/       # Application output samples
└── vosk-model-small-en-us-0.15/  # Speech recognition model
```

## Required Models and Files

- `frozen_inference_graph.pb`: Object detection model weights
- `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Model configuration
- `yolov8n.pt`: YOLOv8 model
- `vosk-model-small-en-us-0.15`: Speech recognition model

## Components

### Machine Learning Models
- **Object Detection**: Using SSD MobileNet v3 and YOLOv8
  - `frozen_inference_graph.pb`
  - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
  - `yolov8n.pt`
  
- **Disease Prediction**:
  - K-Nearest Neighbors (`knnmodeltt.py`)
  - Naive Bayes (`NaiveBayestt.py`)
  - Support Vector Machines for multiple languages
    - Hindi support (`kSVM_Hindi.py`)
    - Kannada support (`kSVME_kannnada.py`)
  
- **Clustering Analysis**:
  - K-means implementation (`k_means_NEMO.py`)
  - K-means evaluation (`k-means-evaluation.py`)

### GUI Applications
- Multilingual Disease Prediction Interface (`newgui_kannada_hindi.py`)
- SVM-based Speech-to-Symptom GUI (`SVM_STS_GUI.py`)

### Data Resources
- `MasterData/`
  - Symptom descriptions (English and Kannada)
  - Precautionary measures
  - Symptom severity data
  - Medical practitioners dataset

### Voice Recognition
- Vosk speech recognition model for English
- Located in `vosk-model-small-en-us-0.15/`

## Setup Requirements

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Required Models:
- Ensure all model files are in the correct locations:
  - Object detection models in `DiseasePred/`
  - Speech recognition model in `vosk-model-small-en-us-0.15/`

## Usage

- For object detection: Run scripts in the `scripts/` directory
- For disease prediction: Use either the command-line tools or GUI applications
- For multilingual support: Use the appropriate language-specific SVM models
- For voice input: Ensure the Vosk model is properly configured

## Documentation

Detailed documentation for individual scripts can be found in their respective directories:
- Script usage instructions: `DiseasePred/scripts/README.md`
- Model specifications: Within the `models/` directory
- Data formats: In the `Data/` and `MasterData/` directories

## Result Samples

Application outputs and screenshots can be found in the `ResultScreenShots/` directory

## Speech Recognition

This project uses the Vosk speech recognition model for English. The model is not included in the repository to reduce size.

### Download Instructions

1. Visit the [Vosk Model Repository](https://alphacephei.com/vosk/models).
2. Download the `vosk-model-small-en-us-0.15` model.
3. Extract the model into the `vosk-model-small-en-us-0.15/` directory at the root of this project.

Ensure the directory structure matches the following:
```
vosk-model-small-en-us-0.15/
├── README
├── am/
├── conf/
├── graph/
└── ivector/
```