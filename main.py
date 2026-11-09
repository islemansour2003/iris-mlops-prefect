import argparse
from pathlib import Path

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "iris_modularisation.csv"
MODEL_PATH = "model.pkl"


def main():
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning Iris")

    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if not any([args.prepare, args.train, args.evaluate, args.all]):
        parser.print_help()
        return

    if args.prepare or args.all:
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        print("Préparation terminée.")
        print(f"Train : {len(X_train)} observations")
        print(f"Test : {len(X_test)} observations")

    if args.train or args.all:
        print("Entraînement du modèle...")
        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model = train_model(X_train, y_train)
        save_model(model, MODEL_PATH)
        print("Entraînement terminé.")

    if args.evaluate or args.all:
        print("Évaluation du modèle...")

        if not Path(MODEL_PATH).exists():
            print("Erreur : modèle introuvable. Exécutez --train d'abord.")
            return

        X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
        model = load_model(MODEL_PATH)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
