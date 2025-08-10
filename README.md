# music-genre-classification-cnn
A CNN-based music genre classification model using the GTZAN dataset and Mel spectrogram features.
## Project Description

This project implements a music genre classification system using both deep learning (CNN) and traditional machine learning models. The goal is to classify audio spectrogram images into 10 music genres. The system involves data preprocessing, feature extraction using HOG, model training, hyperparameter tuning, and deployment of an interactive web application for prediction.
## Setup and Running Locally

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python packages (listed in `requirements.txt`)
### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2 Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3 Install dependencies:
pip install -r requirements.txt

4 Place your dataset folder as described in the instructions (or download it from [dataset source link]).

5 Run the training scripts or load pre-trained models:
python train_cnn.py
python train_traditional_models.py

6 Run the Streamlit app for interactive prediction:
streamlit run streamlit_app.py


