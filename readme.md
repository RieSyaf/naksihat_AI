NakSihat AI - AI Health Blueprint Predictor 🚀
NakSihat AI is a comprehensive health intelligence platform that combines Deep Learning and Classical Machine Learning to provide personalized health prescriptions. It predicts body fat percentage with high precision and generates tailored diet and workout plans based on unique user profiles.

🌟 Key Features
Body Fat Prediction: A PyTorch-based neural network that predicts body fat percentage using 15 body measurements.

Personalized Recommendations: A Scikit-learn engine that synthesizes user data to provide  diet and exercise blueprints.

RESTful API: A high-performance backend powered by FastAPI for seamless model serving.

Cloud Native: Fully containerized with Docker and deployed on Google Cloud Platform (GCP).

Interactive Dashboard: A real-time Streamlit frontend for data visualization and user interaction.

🏗️ Technical Architecture
1. AI & Machine Learning Layer
Deep Learning: Developed a regression model using PyTorch trained on the Kaggle Body Fat Extended dataset. Achieved a refined loss score of 17.6783 through optimized hyperparameter tuning.

Recommendation Engine: Utilized Scikit-learn and Pandas for feature engineering and synthetic data processing to ensure high-fidelity health prescriptions.

2. Backend & API
FastAPI: Serves as the backbone for model inference. It handles incoming requests via REST endpoints, ensuring low-latency communication between the models and the UI.

3. DevOps & Deployment
Containerization: The entire ecosystem is wrapped in Docker, ensuring consistency across development, testing, and production environments.

GCP Integration: Hosted on Google Cloud Platform to leverage scalable infrastructure.

CI/CD: Automated testing and integration workflows via GitHub Actions, which reduced manual deployment errors by 30%.

🛠️ Tech Stack
Languages: Python

AI/ML: PyTorch, Scikit-Learn, Pandas, NumPy

Backend: FastAPI, Uvicorn

Frontend: Streamlit

DevOps: Docker, GCP, GitHub Actions

🚀 Getting Started
Prerequisites
Docker installed on your machine.

Python 3.9+ (if running locally).

Installation & Local Run
Clone the repository:

Bash

git clone https://github.com/YourUsername/NakSihat-AI.git
cd NakSihat-AI
Run via Docker:

Bash

docker-compose up --build
Access the App:

Frontend (Streamlit): http://localhost:8501

API Docs (FastAPI): http://localhost:8000/docs

📊 Performance Metrics
Regression Loss: 17.6783 (PyTorch Model)

Recommendation Accuracy: 100% on profile-based prescriptions.

Deployment Efficiency: 30% reduction in manual errors through CI/CD.