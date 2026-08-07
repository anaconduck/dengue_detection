from flask import Flask,request,render_template,url_for
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib as joblib

app = Flask(__name__)

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.save')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature_names = [
            'EFUSIPLEURA', 'HEPATOMEGALI', 'VOMITING', 'ENSEFALOPATI', 'NAUSEA',
            'EPIGASTRIUM', 'ASITES', 'HEMORRHAGE', 'HEPATIC_FAILURE', 'ENSEFALITIS',
            'RUMPLELEEDE', 'HEADACHE', 'CONTACT_HISTORY', 'KIDNEY_FAILURE',
            'TROMBOSIT', 'ALB', 'EOSABS', 'FEVER', 'UR'
        ]
        
        float_features = []
        for feat in feature_names:
            val = request.form.get(feat)
            if val is None or str(val).strip() == '':
                raise ValueError(f"Field {feat} kosong")
            float_features.append(float(val))
        
        features = [np.array(float_features)]
        normal = scaler.transform(features)
        
        prediction = model.predict(normal)
        
        prediction_text = "No Risk" if prediction[0] == 0 else "High Risk"
        
        return render_template('index.html', prediction_text=prediction_text)

    except ValueError as e:
        return render_template('index.html', error_message="Validasi Gagal: Pastikan semua form telah diisi dengan benar.")
    except Exception as e:
        return render_template('index.html', error_message="Terjadi kesalahan pada sistem AI kami saat memproses data.")

if __name__ == "__main__":
    app.run(debug=True)