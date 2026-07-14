import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Synthetic data (demo)
np.random.seed(42)
n_samples = 500
data = {
    'AGE': np.random.normal(75, 6, n_samples).clip(55, 90),
    'PTGENDER_Male': np.random.binomial(1, 0.55, n_samples),
    'PTEDUCAT': np.random.normal(15.6, 3, n_samples).clip(6, 20),
    'PTRACCAT_White': np.random.binomial(1, 0.8, n_samples),
    'APOE4': np.random.choice([0,1,2], n_samples, p=[0.6,0.3,0.1]),
    'MMSE': np.random.normal(26.9, 3, n_samples).clip(20, 30),
}
df = pd.DataFrame(data)

# Synthetic labels
y = ((df['MMSE'] < 25) | (df['AGE'] > 80) | (df['APOE4'] > 0) | (np.random.random(n_samples) < 0.25)).astype(int)

features = ['PTGENDER_Male', 'PTEDUCAT', 'PTRACCAT_White', 'APOE4']
X = df[features].copy()

# Scale
scaler = MinMaxScaler()
X[['PTEDUCAT']] = scaler.fit_transform(X[['PTEDUCAT']])

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Trained. Score: {model.score(X_train, y_train):.2%}")

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Saved model.joblib & scaler.joblib")

