import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import xgboost as xgb

combined_df = pd.read_csv(r'concatenated.csv')

target_column = 'Diabetes_binary'

combined_df = combined_df.dropna(subset=[target_column])

columns_to_drop = ['Diabetes_012']
combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

X = combined_df.drop(columns=target_column)
y = combined_df[target_column]

print(f"Number of features: {X.shape[1]}")

print(f"Size of the dataset: {X.shape[0]} rows")

if X.shape[0] < 2:
    raise ValueError("Not enough data to split into training and testing sets.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train_resampled)
y_pred_logreg = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg)

linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train_resampled)
y_pred_linear_reg = linear_reg.predict(X_test_scaled)
y_pred_linear_reg_binary = [1 if pred > 0.5 else 0 for pred in y_pred_linear_reg]
accuracy_linear_reg = accuracy_score(y_test, y_pred_linear_reg_binary)
print("Linear Regression Accuracy:", accuracy_linear_reg)

xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)

joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(linear_reg, 'linear_regression_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Models saved successfully.")
