import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "sdn_demo_dataset.csv")
MODEL_PATH = os.path.join(ROOT, "models", "ddos_mlp_model.joblib")
FEATURES = ["flow_duration", "packets_per_second", "bytes_per_second",
            "avg_packet_size", "active_flow_count", "syn_ratio"]

def create_demo_dataset(n=2400, seed=42):
    """Create safe, synthetic flow-statistics rows; no packets are sent."""
    rng = np.random.default_rng(seed)
    n_normal = n // 2
    n_attack = n - n_normal

    normal = np.column_stack([
        rng.uniform(0.2, 20, n_normal),
        rng.normal(85, 35, n_normal).clip(2, 250),
        rng.normal(55000, 22000, n_normal).clip(500, 180000),
        rng.normal(650, 180, n_normal).clip(64, 1500),
        rng.normal(18, 7, n_normal).clip(1, 45),
        rng.beta(2, 8, n_normal)
    ])
    attack = np.column_stack([
        rng.uniform(0.01, 4, n_attack),
        rng.normal(950, 250, n_attack).clip(300, 2200),
        rng.normal(650000, 180000, n_attack).clip(150000, 1400000),
        rng.normal(700, 220, n_attack).clip(64, 1500),
        rng.normal(95, 25, n_attack).clip(40, 180),
        rng.beta(5, 2, n_attack)
    ])
    X = np.vstack([normal, attack])
    y = np.array([0]*n_normal + [1]*n_attack)
    order = rng.permutation(n)
    frame = pd.DataFrame(X[order], columns=FEATURES)
    frame.insert(0, "label", y[order])
    frame["label_name"] = frame["label"].map({0: "Normal", 1: "DDoS-like"})
    return frame

def train_and_save():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df = create_demo_dataset()
    df.to_csv(DATA_PATH, index=False)
    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                          solver="adam", max_iter=350, random_state=42,
                          early_stopping=True, validation_fraction=0.15)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    joblib.dump({"model": model, "scaler": scaler, "metrics": metrics}, MODEL_PATH)
    return metrics

if __name__ == "__main__":
    scores = train_and_save()
    print("Training complete. Evaluation metrics:")
    for name, value in scores.items():
        print(f"{name}: {value}")
