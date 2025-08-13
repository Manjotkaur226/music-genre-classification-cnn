🎵 Music-Genre-Classification-CNN
A CNN-based music genre classification model using the GTZAN dataset and Mel spectrogram features.

📌 Project Description
This project implements a music genre classification system using both deep learning (CNN) and traditional machine learning models.
The goal is to classify audio spectrogram images into 10 music genres.

Pipeline includes:
🔹 Data preprocessing
🔹 Feature extraction using HOG
🔹 Model training
🔹 Hyperparameter tuning
🔹 Deployment of an interactive web application for prediction

⚙️ Setup and Running Locally
Prerequisites
🐍 Python 3.8 or higher
🗂️ Virtual environment (recommended)
📦 Required Python packages (listed in requirements.txt)

Installation Steps
1️⃣ Clone this repository:
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name


2️⃣ Create and activate a virtual environment:

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


3️⃣ Install dependencies:
pip install -r requirements.txt


4️⃣ Place your dataset folder as described in the instructions
(or download it from Kaggle).

5️⃣ Run the training scripts or load pre-trained models:

python train_cnn.py
python train_traditional_models.py


6️⃣ Run the Streamlit app for interactive prediction:
streamlit run streamlit_app.py

🎼 Dataset Information
Dataset Name: GTZAN Music Genre Dataset
Contents: 1000 audio tracks (10 genres, 30 seconds each)
Genres: 🎷 Blues | 🎻 Classical | 🤠 Country | 🎶 Disco | 🎤 HipHop | 🎹 Jazz | 🎸 Metal | 🎵 Pop | 🌴 Reggae | 🎼 Rock
Format: WAV audio files sampled at 22050 Hz

Source: Kaggle
If the dataset is too large to store in this repository:
Download from the link above
Place it in a data/ folder before running the code

🧠 Model Overview & Performance
Model Architecture
🖼️ Input: Mel spectrogram images
🧱 Multiple Conv2D layers (ReLU activation)
📉 MaxPooling layers for downsampling
❌ Dropout layers for regularization
🏗️ Dense layers for classification
🎯 Output layer: Softmax activation

Model Evaluation & Performance Tuning
📊 Evaluated the model on a validation set using accuracy, loss, and confusion matrix
🎨 Applied data augmentation to improve generalization
❌ Added Dropout layers to reduce overfitting
⚡ Tuned learning rate, batch size, and number of epochs
🛠️ Experimented with different CNN architectures and filter sizes

👥 Team Member Roles & Contributions
Member	Role & Contribution
Amita	📂 Dataset selection, data cleaning, cloud development
Anju Bala	📊 Exploratory Data Analysis (EDA), feature engineering
Manjot Kaur	🧠 Model selection, hyperparameter tuning
Krishna Mohandas	💾 Pickle file creation, Streamlit web app
Everyone	🔧 GitHub repository setup and management

🚀 Future Enhancements
🎼 Implement CRNN (Convolutional Recurrent Neural Networks)
🛠️ Apply advanced hyperparameter tuning with Keras Tuner or Optuna
📈 Use larger/more diverse datasets for better generalization

