#TASK ONE TITANIC DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Select relevant features
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# Handle missing values
X.loc[:, 'Age'] = X['Age'].fillna(X['Age'].mean())

# Convert categorical data
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Plot results
labels = ['Correct Predictions', 'Incorrect Predictions']
counts = [(y_test == y_pred).sum(), (y_test != y_pred).sum()]

plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Prediction Type')
plt.ylabel('Count')
plt.title('Model Prediction Accuracy')
plt.show()

# Make a single prediction
test_sample = np.array([[3, 1, 25, 0, 0, 7.25]])  # Example: 3rd class, Male, 25 years old, no family, low fare
prediction = model.predict(test_sample)[0]

print("Single Prediction:", "Survived" if prediction == 1 else "Not Survived")
