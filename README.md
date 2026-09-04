# customer-churn-prediction
End-to-end machine learning project for customer analytics, including churn prediction, fraud detection, customer lifetime value analysis, and personalized recommendations.

---
---

## Features

* Customer Behavior Analysis
* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Customer Churn Prediction
* Customer Lifetime Value (CLV) Analysis
* Fraud Detection
* Recommendation System
* Machine Learning Model Evaluation
* Data Visualization

---

## Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Flask
* HTML
* CSS
* VS Code
* Machine Learning

---

## Project Structure

```text
Customer-Analytics-and-Machine-Learning/
│
├── data/
│   ├── raw/
│   │   ├── customers.xlsx
│   │   ├── transactions.xlsx
│   │   ├── wallet.xlsx
│   │   ├── sessions.xlsx
│   │   └── campaigns.xlsx
│   │
│   └── processed/
│       ├── churn_data.csv
│       ├── clv_data.csv
│       ├── fraud_data.csv
│       └── recommendation_data.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── clv_model.pkl
│   ├── fraud_model.pkl
│   └── recommendation_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_churn.py
│   ├── train_clv.py
│   ├── train_fraud.py
│   └── train_recommendation.py
│
├── templates/
│   ├── index.html
│   ├── churn.html
│   ├── clv.html
│   ├── fraud.html
│   └── recommendation.html
│
├── static/
│   └── css/
│       └── style.css
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd Customer-Analytics-and-Machine-Learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the Flask application:

```bash
python app.py
```

After starting the application, open the local Flask address shown in the terminal in your web browser.

---

## Usage

1. Start the Flask application.
2. Open the application in a web browser.
3. Select the required customer analytics module.
4. Provide the required input data.
5. Submit the input for prediction or analysis.
6. View the generated result.

The application provides separate functionality for:

* Customer Churn Prediction
* Customer Lifetime Value Analysis
* Fraud Detection
* Recommendation System

---

## Data Preprocessing

The project performs data preprocessing before applying machine learning models.

The preprocessing steps include:

* Handling missing values
* Removing duplicate records
* Data type conversion
* Combining relevant datasets
* Selecting relevant features
* Preparing data for machine learning

---

## Exploratory Data Analysis

Exploratory Data Analysis is performed to understand customer behavior and identify useful patterns in the data.

The analysis includes:

* Customer behavior analysis
* Transaction analysis
* Purchase patterns
* Session activity
* Campaign analysis
* Correlation analysis
* Data visualization

---

## Feature Engineering

Feature engineering is performed to create meaningful features from customer, transaction, wallet, session, and campaign data.

The engineered features are used to prepare the datasets for different machine learning tasks and improve model performance.

---

## Machine Learning Models

The project includes four major machine learning applications:

### Customer Churn Prediction

Predicts whether a customer is likely to leave based on customer behavior and activity.

### Customer Lifetime Value

Analyzes and predicts the potential value of a customer based on available customer and transaction information.

### Fraud Detection

Identifies potentially fraudulent transactions using transaction-related features.

### Recommendation System

Generates recommendations based on customer and transaction behavior.

---

## Model Evaluation

The trained models are evaluated using appropriate machine learning evaluation metrics.

The evaluation results are used to measure model performance and compare prediction effectiveness.

---

## Known Limitations

* Model performance depends on the quality and quantity of available data.
* Prediction results may vary depending on the selected features and machine learning algorithm.
* The application is designed and tested in a local environment.
* The models may require retraining when new datasets are introduced.

---

## Future Improvements

* Improve model accuracy using advanced machine learning algorithms.
* Add real-time data processing.
* Improve the recommendation system.
* Deploy the application on a cloud platform.
* Add an interactive analytics dashboard.
* Automate model retraining with new data.

---

## Acknowledgments

This project was developed as a **Customer Analytics and Machine Learning project** using Python, machine learning, and Flask to analyze customer-related data and provide useful predictions and insights.

