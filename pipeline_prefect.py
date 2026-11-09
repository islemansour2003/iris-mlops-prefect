import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MAIN_PY = BASE_DIR / "main.py"
REQUIREMENTS = BASE_DIR / "requirements.txt"
MODEL_PATH = BASE_DIR / "model.pkl"

CODE_FILES = [
    str(BASE_DIR / "model_pipeline.py"),
    str(BASE_DIR / "main.py"),
    str(BASE_DIR / "pipeline_prefect.py"),
]


# ==========================================
# FONCTION UTILITAIRE
# ==========================================


def run_main(argument):
    subprocess.run(
        [sys.executable, str(MAIN_PY), argument],
        check=True,
        cwd=BASE_DIR,
    )


# ==========================================
# TASK 1 : INSTALLATION
# ==========================================


@task(name="install-dependencies")
def install_dependencies():
    logger = get_run_logger()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS),
        ],
        check=True,
        cwd=BASE_DIR,
    )

    logger.info("Dependencies installed successfully")


# ==========================================
# TASK 2 : PREPARATION
# ==========================================


@task(name="prepare-data")
def prepare_data():
    logger = get_run_logger()

    run_main("--prepare")

    logger.info("Data preparation completed")


# ==========================================
# TASK 3 : ENTRAINEMENT
# ==========================================


@task(name="train-model")
def train_model():
    logger = get_run_logger()

    run_main("--train")

    logger.info("Model training completed")


# ==========================================
# TASK 4 : SAUVEGARDE
# ==========================================


@task(name="save-model")
def save_model():
    logger = get_run_logger()

    # main.py --train sauvegarde deja model.pkl
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    logger.info("Model saved successfully: %s", MODEL_PATH)


# ==========================================
# TASK 5 : CHARGEMENT
# ==========================================


@task(name="load-model")
def load_model():
    logger = get_run_logger()

    # Verifier le modele sauvegarde
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    logger.info("Model file available: %s", MODEL_PATH)


# ==========================================
# TASK 6 : EVALUATION
# ==========================================


@task(name="evaluate-model")
def evaluate_model():
    logger = get_run_logger()

    run_main("--evaluate")

    logger.info("Model evaluation completed")


# ==========================================
# TASK 7 : FORMATAGE BLACK
# ==========================================


@task(name="format-code")
def format_code():
    logger = get_run_logger()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            *CODE_FILES,
        ],
        check=True,
        cwd=BASE_DIR,
    )

    logger.info("Code formatting completed")


# ==========================================
# TASK 8 : QUALITE FLAKE8
# ==========================================


@task(name="check-code-quality")
def check_code_quality():
    logger = get_run_logger()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            "--extend-ignore=E203,W503",
            *CODE_FILES,
        ],
        check=True,
        cwd=BASE_DIR,
    )

    logger.info("Code quality check completed")


@task(name="check-security")
def check_security():
    logger = get_run_logger()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-ll",
            *CODE_FILES,
        ],
        check=True,
        cwd=BASE_DIR,
    )

    logger.info("Security check completed")


@task(name="run-tests")
def run_tests():
    logger = get_run_logger()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "test_pipeline.py",
        ],
        check=True,
        cwd=BASE_DIR,
    )

    logger.info("Unit tests completed")


# ==========================================
# FLOWS
# ==========================================


@flow(name="all")
def flow_all():
    install_dependencies()
    flow_code()
    prepare_data()
    train_model()
    save_model()
    evaluate_model()


@flow(name="train")
def flow_train():
    prepare_data()
    train_model()


@flow(name="evaluate")
def flow_evaluate():
    load_model()
    evaluate_model()


@flow(name="code")
def flow_code():
    format_code()
    check_code_quality()
    check_security()
    run_tests()


# ==========================================
# DICTIONNAIRE DES FLOWS
# ==========================================

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}


# ==========================================
# EXECUTION AVEC ARGPARSE
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect")

    parser.add_argument(
        "--flow",
        choices=FLOWS.keys(),
        required=True,
        help="Flow a executer",
    )

    args = parser.parse_args()

    FLOWS[args.flow]()
