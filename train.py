import os
import sys
import pickle
import numpy as np

# Path setup taake 'src' folder mil jaye
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE) 

# Modular imports
from src.models import get_knn_model, get_dt_model, get_kmeans_model
from src.evaluator import save_confusion_matrix
from src.data_loader import load_dataset

# 1. Data load karein (Ye 'loan_data.csv' create bhi karega aur load bhi)
print("Loading dataset...")
X_train, X_test, y_train, y_test = load_dataset('loan_data.csv')

# static folder banayein agar nahi hai
if not os.path.exists('static'):
    os.makedirs('static')

# 2. Train and Save KNN (Accuracy target: 64.07%)
print("Training KNN...")
knn = get_knn_model()
knn.fit(X_train, y_train)
pickle.dump(knn, open('static/knn.pkl', 'wb'))
save_confusion_matrix(y_test, knn.predict(X_test), "KNN", "knn")

# 3. Train and Save Decision Tree (Accuracy target: 82.30%)
print("Training Decision Tree...")
dt = get_dt_model()
dt.fit(X_train, y_train)
pickle.dump(dt, open('static/dt.pkl', 'wb'))
save_confusion_matrix(y_test, dt.predict(X_test), "Decision Tree", "dt")

# 4. Train and Save K-Means (Accuracy target: 68.64%)
print("Training K-Means...")
kmeans = get_kmeans_model()
kmeans.fit(X_train) # Unsupervised learning
pickle.dump(kmeans, open('static/kmeans.pkl', 'wb'))
# K-means ka confusion matrix banane ke liye cluster labels ko use karte hain
save_confusion_matrix(y_train, kmeans.labels_, "K-Means", "kmeans")

print("\nSuccess! All models and charts generated in 'static/' folder.")