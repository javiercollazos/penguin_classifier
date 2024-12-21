import requests
import json
import time

def make_prediction(model_name, data):
    url = f'http://127.0.0.1:5000/predict/{model_name}'
    print(f"\nEnviando datos al modelo {model_name}:")
    print(json.dumps(data, indent=2))
    
    try:
        response = requests.post(url, json=data)
        print(f"Código de respuesta HTTP: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error en la respuesta del servidor: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error en la petición: {e}")
        return None

# Datos de ejemplo del pingüino con todas las características necesarias
penguin_data = {
    'bill_length_mm': 45.2,
    'bill_depth_mm': 15.8,
    'flipper_length_mm': 215,
    'body_mass_g': 5200,
    'island_Biscoe': 1,
    'island_Dream': 0,
    'island_Torgersen': 0,
    'sex_female': 0,
    'sex_male': 1
}

# Lista de modelos disponibles
models = ['logistic_regression', 'svm', 'decision_tree', 'knn']

print("=== Iniciando pruebas de clasificación de pingüinos ===")
print("\nDatos del pingüino a clasificar:")
print(json.dumps(penguin_data, indent=2))

for model in models:
    print(f"\nModelo: {model}")
    print("-" * 50)
    
    for i in range(2):
        print(f"\nPetición {i+1}:")
        result = make_prediction(model, penguin_data)
        
        if result is not None:
            if 'error' in result:
                print(f"Error del servidor: {result['error']}")
            else:
                print("Resultado de la predicción:")
                print(f"Especie predicha: {result['prediction']}")
                print(f"Probabilidad: {result['probability']:.4f}")
        else:
            print("No se pudo obtener predicción")
        
        time.sleep(1)

print("\n=== Proceso de clasificación completado ===")


