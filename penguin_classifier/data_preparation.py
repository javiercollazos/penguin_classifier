import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data():
    # Cargar datos
    df = sns.load_dataset("penguins")
    
    # Eliminar filas con valores NA
    df = df.dropna()
    
    # Separar características y etiquetas
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Codificar variables categóricas
    X = pd.get_dummies(X, drop_first=False)  # Cambiado a False para mantener todas las columnas
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar variables numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data()
    print("Datos preparados correctamente.")
