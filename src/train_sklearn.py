import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_from_csv(csv_path="data/processed/features.csv",
                   model_out="models/sklearn_model.pkl"):
    """
    Train a simple ML model (Logistic Regression) from extracted audio features
    and save it along with the scaler and label encoder.
    """
    csv_path = Path(csv_path)
    model_out = Path(model_out)

    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {csv_path}. Run dataset_builder.py first.")

    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset from {csv_path}, shape = {df.shape}")

    # Features = f0...fn, Labels = genre
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (safe against small datasets)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.33, random_state=42, stratify=y_encoded
        )
    except ValueError:
        print("⚠️ Not enough samples per class to stratify. Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.33, random_state=42
        )

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model, scaler, and label encoder
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "label_encoder": le}, model_out)
    print(f"💾 Saved trained model to {model_out}")

    return model, le, scaler


if __name__ == "__main__":
    train_from_csv()
