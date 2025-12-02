# ANNClassificationChurn
A deep learning project that builds an Artificial Neural Network (ANN) to predict customer churn based on demographic and account-related features. The model is trained on a structured dataset and uses several preprocessing steps such as encoding categorical variables, feature scaling, and splitting the dataset into training and testing sets.

🚀 Features

ANN model built with TensorFlow/Keras

Preprocessing: encoding + scaling

Saved model and artifacts (model.h5, scaler.pkl, onehotEncoderGeo.pkl,labelEncoderGender.pkl)

Script for predicting new customer churn

Easy to deploy in Flask / FastAPI / Streamlit / APEX

🏗️ ANN Model Architecture

Input Layer

Hidden Layer 1 — Dense + ReLU

Hidden Layer 2 — Dense + ReLU

Output Layer — Dense + Sigmoid

Loss: Binary Crossentropy

Optimizer: Adam

Metrics: Accuracy

📈 Model Performance

Accuracy: ~86–88%

Validation accuracy: ~85–87%

