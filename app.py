from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Models load karein
knn = pickle.load(open('static/knn.pkl', 'rb'))
dt = pickle.load(open('static/dt.pkl', 'rb'))
km = pickle.load(open('static/kmeans.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Input lena aur Error se bachne ke liye reshape karna
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1) 
    
    # Predictions
    res_knn = "Approved ✅" if knn.predict(final_features)[0] == 1 else "Rejected ❌"
    res_dt = "Approved ✅" if dt.predict(final_features)[0] == 1 else "Rejected ❌"
    res_km = "Approved ✅" if km.predict(final_features)[0] == 1 else "Rejected ❌"

    comparison = {
        'knn': {'res': res_knn, 'acc': '64.07%', 'img': 'cm_knn.png'},
        'dt':  {'res': res_dt,  'acc': '82.30%', 'img': 'cm_dt.png'},
        'km':  {'res': res_km,  'acc': '68.64%', 'img': 'cm_kmeans.png'}
    }
    
    return render_template('index.html', data=comparison)

if __name__ == "__main__":
    app.run(debug=True)