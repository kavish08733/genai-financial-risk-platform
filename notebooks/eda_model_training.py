# GenAI-Enhanced Financial Risk Platform
# Portfolio-Grade AI Credit Risk Assessment with Visualizations, Explainability, and Business Insights

# 1. Data Loading & Overview
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('figures', exist_ok=True)

# For feature importance and explainability
from sklearn.inspection import permutation_importance
import shap

# For model evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load processed data
df = pd.read_csv('sample_data/uci_credit_processed.csv')
print("Data Sample:")
print(df.head())

# 2. EDA: Class Balance, Feature Distributions, Outlier Detection
print("\nClass Balance:")
print(df['default'].value_counts(normalize=True))
sns.countplot(x='default', data=df)
plt.title('Class Balance: Default Payment Next Month')
plt.savefig('figures/class_balance.png', bbox_inches='tight')
plt.show()

# Feature distributions (histograms)
num_features = ['LIMIT_BAL', 'AGE', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT']
for col in num_features:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution: {col}')
    plt.savefig(f'figures/feature_distribution_{col}.png', bbox_inches='tight')
    plt.show()

# Boxplots for outlier detection
for col in num_features:
    sns.boxplot(x='default', y=col, data=df)
    plt.title(f'Boxplot: {col} by Default')
    plt.savefig(f'figures/boxplot_{col}_by_default.png', bbox_inches='tight')
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('figures/feature_correlation_heatmap.png', bbox_inches='tight')
plt.show()

# 3. Data Preprocessing
print("\nPreprocessing and train/test split...")
target = 'default'
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (Neural Network)
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=2)

# 5. Model Performance & Visualizations
y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)
print(f'\nAccuracy: {acc:.4f}')
print(f'ROC-AUC: {roc:.4f}')
print('Confusion Matrix:\n', cm)

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('figures/confusion_matrix.png', bbox_inches='tight')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC Curve (AUC = %.2f)' % roc)
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('figures/roc_curve.png', bbox_inches='tight')
plt.show()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(rec, prec, label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('figures/precision_recall_curve.png', bbox_inches='tight')
plt.show()

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Feature Importance (Permutation Importance)
print("\nCalculating permutation feature importance...")
perm = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=42, scoring='roc_auc')
importances = pd.Series(perm.importances_mean, index=features)
importances = importances.sort_values(ascending=False)
print(importances.head(10))
importances.head(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (Permutation)')
plt.gca().invert_yaxis()
plt.savefig('figures/feature_importance_permutation.png', bbox_inches='tight')
plt.show()

# 7. SHAP Explainability (Global & Local)
print("\nCalculating SHAP values...")
explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled[:10], nsamples=100)
shap.summary_plot(shap_values, X_test.iloc[:10], feature_names=features, show=False)
plt.title('SHAP Summary Plot (Sample)')
plt.savefig('figures/shap_summary_plot_sample.png', bbox_inches='tight')
plt.show()

# 8. Hybrid Scoring: Rule-Based + Neural Network
# Example rule: If PAY_0 >= 2 (serious recent delay) or LIMIT_BAL < 50000, flag as high risk
def rule_based_score(row):
    if row['PAY_0'] >= 2 or row['LIMIT_BAL'] < 50000:
        return 1
    return 0

X_test_eval = X_test.copy()
X_test_eval['nn_pred'] = y_pred
X_test_eval['rule_pred'] = X_test.apply(rule_based_score, axis=1)
# Hybrid: flag as high risk if either model or rule says so
X_test_eval['hybrid_pred'] = ((X_test_eval['nn_pred'] + X_test_eval['rule_pred']) > 0).astype(int)

print('Hybrid Model Classification Report:')
print(classification_report(y_test, X_test_eval['hybrid_pred']))

# 9. Batch Prediction Output & Top N Insights
X_test_eval['prob'] = y_pred_prob
print("\nTop 5 High-Risk Predictions:")
print(X_test_eval.sort_values('prob', ascending=False).head(5))
print("\nTop 5 Low-Risk Predictions:")
print(X_test_eval.sort_values('prob', ascending=True).head(5))

# Visualize prediction probabilities
sns.histplot(X_test_eval['prob'], bins=30, kde=True)
plt.title('Distribution of Predicted Default Probabilities')
plt.xlabel('Predicted Probability of Default')
plt.savefig('figures/predicted_default_probabilities_distribution.png', bbox_inches='tight')
plt.show()

# 10. (Optional) LangChain/OpenAI Text Explanations
# Uncomment and configure your OpenAI API key to use this section
# from langchain.chains.llm import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# 
# template = PromptTemplate.from_template(
#     'Explain why a financial risk model would classify someone with credit score {score} and PAY_0 {pay0} as high risk.'
# )
# llm = OpenAI(temperature=0.3)
# chain = LLMChain(llm=llm, prompt=template)
# # Example: explain a real prediction
# sample = X_test.iloc[0]
# explanation = chain.run({'score': sample.get('BILL_AMT1', 0), 'pay0': sample['PAY_0']})
# print('Model Explanation:', explanation)

# 11. Cumulative Gains/Lift Chart
# Sort by predicted probability
import numpy as np
sorted_idx = np.argsort(-X_test_eval['prob'].values)
cum_defaults = np.cumsum(y_test.values[sorted_idx])
total_defaults = y_test.sum()
percent_customers = np.arange(1, len(y_test)+1) / len(y_test)
percent_defaults = cum_defaults / total_defaults
plt.plot(percent_customers, percent_defaults, label='Model')
plt.plot([0,1],[0,1],'--',color='gray', label='Random')
plt.xlabel('Proportion of Customers Targeted')
plt.ylabel('Proportion of Defaulters Captured')
plt.title('Cumulative Gains Chart')
plt.legend()
plt.savefig('figures/cumulative_gains_chart.png', bbox_inches='tight')
plt.show()

# 12. SHAP Force Plot for Highest-Risk Customer
# (If running in Jupyter, this will be interactive. In script, fallback to summary plot if needed.)
try:
    shap.initjs()
    idx = X_test_eval['prob'].idxmax()
    explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])
    shap_values = explainer.shap_values(X_test_scaled[[list(X_test_eval.index).index(idx)]], nsamples=100)
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[[list(X_test_eval.index).index(idx)]], matplotlib=True, show=True)
except Exception as e:
    print('SHAP force plot could not be rendered, showing summary plot instead.')
    shap.summary_plot(shap_values, X_test.iloc[:10], feature_names=features, show=False)
    plt.title('SHAP Summary Plot (Sample)')
    plt.savefig('figures/shap_summary_plot_sample.png', bbox_inches='tight')
    plt.show()

# 13. Top N At-Risk Customers Table
N = 10
print(f"\nTop {N} At-Risk Customers (by predicted probability):")
top_n = X_test_eval.sort_values('prob', ascending=False).head(N)
print(top_n[['prob', 'nn_pred', 'rule_pred', 'hybrid_pred', 'LIMIT_BAL', 'AGE', 'PAY_0', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT']])

# ---
# This script demonstrates a full, business-ready workflow for AI-powered financial risk modeling, hybrid scoring, and explainability, ready for portfolio and real-world use. 