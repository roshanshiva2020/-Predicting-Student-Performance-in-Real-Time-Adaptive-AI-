import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset
data = {
    'study_hours': [2, 4, 5, 1, 7, 8, 6, 3],
    'attendance': [60, 80, 90, 50, 95, 98, 85, 65],
    'previous_scores': [50, 65, 70, 40, 88, 92, 75, 55],
    'online_activity': [30, 60, 70, 20, 80, 90, 75, 40],
    'sleep_hours': [5, 6, 7, 4, 8, 7, 6, 5],
    'performance': [0, 1, 1, 0, 1, 1, 1, 0]  # 1=Pass, 0=Fail
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('performance', axis=1)
y = df['performance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example new student data (real-time adaptation)
new_student = [[6, 85, 78, 70, 7]]  # features
prediction = model.predict(new_student)
print("Prediction for new student (1=Pass, 0=Fail):", prediction[0])
