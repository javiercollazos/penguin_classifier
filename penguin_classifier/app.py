from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("Iniciando carga de modelos...")

try:
    models = {
        'logistic_regression': joblib.load('logistic_regression_model.joblib'),
        'svm': joblib.load('svm_model.joblib'),
        'decision_tree': joblib.load('decision_tree_model.joblib'),
        'knn': joblib.load('knn_model.joblib')
    }
    print("Modelos cargados correctamente")

    scaler = joblib.load('scaler.joblib')
    print("Scaler cargado correctamente")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    raise

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Ordenar características en el mismo orden que durante el entrenamiento
        features = [
            data['bill_length_mm'],
            data['bill_depth_mm'],
            data['flipper_length_mm'],
            data['body_mass_g'],
            data['island_Biscoe'],
            data['island_Dream'],
            data['island_Torgersen'],
            data['sex_female'],
            data['sex_male']
        ]
        
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        
        model = models[model_name]
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features).max()
        
        return jsonify({
            'prediction': prediction[0],
            'probability': float(probability)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
