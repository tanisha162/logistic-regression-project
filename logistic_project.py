import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data: hours studied vs pass/fail (1 = pass, 0 = fail)
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":        [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print(df)

# Features and label
X = df[["hours_studied"]]
y = df["passed"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check what it learned
print("\nWeight:", model.coef_)
print("Bias:", model.intercept_)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nPredictions:", predictions)
print("Probabilities:\n", probabilities)
print("Actual:", y_test.values)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

# Try predicting for a brand new student
new_student = [[4.5]]   # studied 4.5 hours
new_prediction = model.predict(new_student)
new_probability = model.predict_proba(new_student)
print("\nNew student (4.5 hrs) - Predicted class:", new_prediction)
print("New student (4.5 hrs) - Probability:", new_probability)

# Save results to CSV
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions,
    "Probability_of_Pass": probabilities[:, 1]
})
results_df.to_csv("results.csv", index=False)
print("\nSaved results.csv")