from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load models
models = {}
for model_name in ['logistic_regression_model', 'decision_tree_model', 'random_forest_model', 'support_vector_machine_model', 'naive_bayes_model']:
    with open(f'{model_name}.pkl', 'rb') as f:
        models[model_name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    cibil_score = float(data['cibil_score'])
    loan_term = float(data['loan_term'])
    income = float(data['income'])
    loan_amount = float(data['loan_amount'])
    
    income_to_loan_ratio = income / loan_amount
    input_df = pd.DataFrame([[cibil_score, loan_term, income_to_loan_ratio]], columns=['cibil_score', 'loan_term', 'income_to_loan_ratio'])
    input_scaled = scaler.transform(input_df)
    
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(input_scaled)
        status = 'Rejected' if prediction[0] else 'Approved'
        predictions[name.replace('_model', '').replace('_', ' ').title()] = status

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
