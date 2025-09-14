import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_from_csv(csv_path, model_out='rf_model.joblib'):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['filename','label']).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    joblib.dump(clf, model_out)
    print('Saved model to', model_out)

if __name__ == '__main__':
    train_from_csv('features.csv')
