# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 2: Load the Dataset
iris_data = datasets.load_iris()
features = iris_data.data  # Feature matrix (sepal & petal measurements)
labels = iris_data.target  # Target labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Convert to DataFrame for better visualization
iris_df = pd.DataFrame(features, columns=iris_data.feature_names)
iris_df['flower_type'] = labels

# Step 3: Exploratory Data Analysis
print("First 5 rows of dataset:\n", iris_df.head())
print("\nDataset Overview:\n", iris_df.describe())

# Visualizing feature relationships
sns.pairplot(iris_df, hue="flower_type", diag_kind="kde")
plt.show()

# Step 4: Data Preprocessing
# Splitting dataset into 80% training and 20% testing
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardizing the dataset
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Step 5: Train the Model (Random Forest Classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(train_features, train_labels)

# Step 6: Model Evaluation
predictions = classifier.predict(test_features)
model_accuracy = accuracy_score(test_labels, predictions)
print("\nModel Accuracy:", model_accuracy)
print("\nDetailed Report:\n", classification_report(test_labels, predictions))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(test_labels, predictions), annot=True, cmap="Blues", fmt="d",
            xticklabels=iris_data.target_names, yticklabels=iris_data.target_names)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()

# Step 7: Making Predictions
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example measurement
sample_transformed = scaler.transform(sample_input)
predicted_species = classifier.predict(sample_transformed)
print("\nPredicted Iris Species:", iris_data.target_names[predicted_species[0]])
