# Customer Churn Analysis

An end-to-end beginner-friendly project to explore subscription cancellations, customer engagement, and retention opportunities using Python, Pandas, Matplotlib, Streamlit, and Scikit-learn.

## Features
- Customer churn and retention KPIs
- Churn distribution and churn by contract visualizations
- Behaviour analysis for usage, tenure, satisfaction, and support tickets
- Baseline Logistic Regression churn classifier
- Accuracy, precision, recall, F1 score, and confusion matrix
- Practical customer-retention recommendations
- Optional upload of your own CSV dataset

## Project structure
```
customer-churn-analysis/
├── app.py
├── customer_churn.csv
├── requirements.txt
└── README.md
```

## Setup (Windows / VS Code)
1. Install Python 3.10 or newer.
2. Clone or download this repository and open its folder in VS Code.
3. In the terminal run:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL displayed in the terminal.

## Dataset
`customer_churn.csv` contains 1,200 simulated customer records. The target column is `churn` (1 = churned, 0 = retained). Features include tenure, monthly charges, weekly usage, support tickets, satisfaction, contract type, and payment method. This synthetic dataset is for learning and is not real customer information.

## Model
The app trains a Logistic Regression baseline with preprocessing for numeric and categorical columns. It reports classification metrics on a train/test split. Validate performance on representative, authorized data before any business use.

## Tutorial
Project reference video: https://youtu.be/6EmjRXUcARc

## License
For educational use. Add a license if you plan to distribute or reuse the project.
