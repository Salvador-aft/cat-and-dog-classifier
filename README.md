# Cat and Dog Classifier

## Description
Image classifier for cats and dogs using TensorFlow/Keras with FastAPI backend and DVC for model storage.

## Features
- Image classification with pretrained model
- Web interface for image upload
- Model versioning with DVC and AWS S3

## Installation

### Requirements
- Python 3.8+
- Git
- DVC with AWS credentials

### Setup Steps
1. Clone repository:
   ```bash
   git clone https://github.com/Salvador-aft/cat-and-dog-classifier.git
   cd cat-and-dog-classifier

2. Create and activate virtual environment:
    python -m venv venv
    source venv/bin/activate // Linux/Mac
    venv\Scripts\activate // Windows

3. Install dependencies:
    pip install -r requirements.txt

4. Download model data:
    dvc pull

# Project Structure
    .
    ├── app.py                // FastAPI backend
    ├── cat_dog_classifier.h5 // Model (managed by DVC)
    ├── index.html            // Frontend
    ├── requirements.txt      // Dependencies
    └── train_model.py        // Training script

## Usage

### Running the Application
    uvicorn app:app --reload
Access the interface at http://localhost:8000

### Training the Model
    python train_model.py

## Data Management
Model stored in AWS S3 (bucket: cat-dog-classifier-models)
Configure DVC:
    dvc remote add -d myremote s3://cat-dog-classifier-models/

## License
MIT License