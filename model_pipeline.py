import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(file_path):
    # 1. Charger le dataset
    df = pd.read_csv(file_path)

    # 2. Séparer les caractéristiques et la cible
    X = df.drop("species", axis=1)
    y = df["species"]

    # 3. Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Retourner les données
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    # 1. Créer le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 2. Entraîner le modèle
    model.fit(X_train, y_train)

    # 3. Retourner le modèle entraîné
    return model


def evaluate_model(model, X_test, y_test):
    # Prédire les classes
    y_pred = model.predict(X_test)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Afficher les résultats
    print(f"Accuracy : {accuracy:.4f}")
    print("Classification Report :")
    print(classification_report(y_test, y_pred))

    return accuracy


def save_model(model, file_path="model.pkl"):
    # Sauvegarder le modèle entraîné
    joblib.dump(model, file_path)

    print(f"Modèle sauvegardé dans : {file_path}")


def load_model(file_path="model.pkl"):
    # Charger le modèle sauvegardé
    model = joblib.load(file_path)

    print(f"Modèle chargé depuis : {file_path}")

    return model
