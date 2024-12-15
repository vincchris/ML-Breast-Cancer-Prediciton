import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_logistic_regression_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Logistic Regression Model Accuracy: ', accuracy_score(y_test, y_pred))
    print('Logistic Regression Classification Report: \n', classification_report(y_test, y_pred))

    return model, scaler


def create_random_forest_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Random Forest Model Accuracy: ', accuracy_score(y_test, y_pred))
    print('Random Forest Classification Report: \n', classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    """
    Load and clean breast cancer dataset.
    """
    data = pd.read_csv("../data/data.csv")

    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Map 'diagnosis' column to binary (1 for Malignant, 0 for Benign)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    # Logistic Regression
    logistic_model, logistic_scaler = create_logistic_regression_model(data)
    with open('../model/logistic_model.pkl', 'wb') as f:
        pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../model/logistic_scaler.pkl', 'wb') as f:
        pickle.dump(logistic_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Random Forest
    random_forest_model, random_forest_scaler = create_random_forest_model(data)
    with open('../model/random_forest_model.pkl', 'wb') as f:
        pickle.dump(random_forest_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../model/random_forest_scaler.pkl', 'wb') as f:
        pickle.dump(random_forest_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
