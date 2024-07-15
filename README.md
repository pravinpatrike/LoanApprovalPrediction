# Loan-Approval-Prediction-Using-Machine-Learning
## Objective:
The objective of this project is to build a predictive model that can accurately determine whether a loan application will be approved or rejected. This will help financial institutions streamline their loan approval process, reduce manual effort, and improve decision-making accuracy. It includes data preparation, model training, hyperparameter tuning, and deployment through a Flask web application.
## Process Flow:
The project covers data collection, data cleaning, EDA, feature engineering, model training, evaluation, and deployment via a web application.
- Data Collection
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Model Building
- Model Selection
- Model Evaluation
- Model Deployment
## Usage:
**Model Training**: 

To train the models, run: python train_models.py

This will:
- Load and preprocess the dataset.
- Split the data into training and testing sets.
- Train multiple models with hyperparameter tuning.
- Save the best models and scaler for future use.

**Web Application**

To start the web application, run: python app.3.0.py

Open your web browser and navigate to http://127.0.0.1:5000 to access the application.
## Web Application

The Flask web application allows users to input loan details and get predictions from the trained models. The application includes:
- A home page for inputting loan details.
- A results page showing predictions from all models.
## Results and Conclusion
The trained models are evaluated based on accuracy, precision, recall, and F1-score and acheived *99.6%* accuracy. The results are summarized and displayed in the web application. The project successfully showcases the end-to-end process of building, training, and deploying machine learning models for a practical application. The web application provides a user-friendly platform for making loan approval predictions, demonstrating the potential of machine learning in financial decision-making.

By integrating a predictive loan approval model, financial institutions can transform their loan processing operations, leading to improved efficiency, cost savings, and better risk management. This model not only supports the institution's business goals but also enhances customer satisfaction through faster and more transparent loan application processing.



