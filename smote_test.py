import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

combined_df = pd.read_csv('concatenated.csv')

combined_df = combined_df.dropna(subset=['Diabetes_binary'])
combined_df = combined_df.drop(columns=['Diabetes_012'], errors='ignore')

X = combined_df.drop(columns='Diabetes_binary')
y = combined_df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)

smote = SMOTE(random_state=42, k_neighbors=nn)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("SMOTE operation completed successfully.")
