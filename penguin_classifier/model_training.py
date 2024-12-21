from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preparation import prepare_data

def train_models():
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data()
    
    models = {
        'logistic_regression': LogisticRegression(),
        'svm': SVC(probability=True),
        'decision_tree': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy de {name}: {accuracy:.4f}")
        joblib.dump(model, f'{name}_model.joblib')
    
    # Guardar el scaler
    joblib.dump(scaler, 'scaler.joblib')

if __name__ == "__main__":
    train_models()
    print("Modelos entrenados y guardados.")


