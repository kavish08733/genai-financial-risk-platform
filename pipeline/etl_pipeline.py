import pandas as pd
import numpy as np

def extract():
    # Read the UCI Default of Credit Card Clients dataset (header=1 skips the first row of description)
    df = pd.read_excel('sample_data/default of credit card clients.xls', header=1)
    return df

def transform(df):
    # Rename target column for convenience
    df = df.rename(columns={"default payment next month": "default"})
    # Drop ID column
    df = df.drop(columns=["ID"])
    # Encode categorical variables (SEX, EDUCATION, MARRIAGE)
    df['SEX'] = df['SEX'].map({1: 'male', 2: 'female'})
    df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
    # Optionally, create new features (e.g., total bill, total payment)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['TOTAL_BILL_AMT'] = df[bill_cols].sum(axis=1)
    df['TOTAL_PAY_AMT'] = df[pay_cols].sum(axis=1)
    # Fill any missing values with 0 (dataset should not have missing, but for robustness)
    df = df.fillna(0)
    return df

def load(df):
    df.to_csv('sample_data/uci_credit_processed.csv', index=False)

if __name__ == "__main__":
    data = extract()
    transformed = transform(data)
    load(transformed)
