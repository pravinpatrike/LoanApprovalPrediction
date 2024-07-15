# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
loan_data = pd.read_csv('loan_approval_dataset.csv')
loan_data.columns = loan_data.columns.str.strip()
# Creating a new feature: income_to_loan_ratio
loan_data['income_to_loan_ratio'] = loan_data['income_annum'] / loan_data['loan_amount']
loan_data_encoded = pd.get_dummies(loan_data, drop_first=True)

# Select features and target
top_features = ['cibil_score', 'loan_term','income_to_loan_ratio']
X = loan_data_encoded[top_features]
y = loan_data_encoded['loan_status_ Rejected']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define models and hyperparameters for tuning
models = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 10, 20],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 5]
        }
    },
    'Support Vector Machine': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {}  # GaussianNB has no hyperparameters to tune
    }
}

# Train and save the best models
best_models = {}
for name, config in models.items():
    clf = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    best_models[name] = clf.best_estimator_
    with open(f'{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
        pickle.dump(clf.best_estimator_, f)

# Print the results
results = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    })

results_df = pd.DataFrame(results)
print("\nModel Performance with best Hyper Parameters:")
print(results_df)
