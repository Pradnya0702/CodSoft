# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 2: Download and Load Dataset
#dataset_url = "https://raw.githubusercontent.com/nikhilsingh097/advertising_dataset/main/Advertising.csv"
sales_data = pd.read_csv("/content/advertising.csv")

# Step 3: Explore the Dataset
print("\nFirst 5 rows of dataset:\n", sales_data.head())
print("\nDataset Info:\n", sales_data.info())
print("\nStatistical Summary:\n", sales_data.describe())

# Check for missing values
print("\nMissing Values:\n", sales_data.isnull().sum())

# Step 4: Data Visualization
sns.pairplot(sales_data)
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(sales_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Step 5: Data Preprocessing
features = sales_data[['TV', 'Radio', 'Newspaper']]  # Independent variables
target_variable = sales_data['Sales']  # Dependent variable

# Splitting data into Training and Testing sets (80%-20%)
features_train, features_test, target_train, target_test = train_test_split(
    features, target_variable, test_size=0.2, random_state=42
)

# Scaling features
scaler_obj = StandardScaler()
features_train = scaler_obj.fit_transform(features_train)
features_test = scaler_obj.transform(features_test)

# Step 6: Train the Model
regression_model = LinearRegression()
regression_model.fit(features_train, target_train)

# Step 7: Evaluate the Model
target_predictions = regression_model.predict(features_test)
mae_value = mean_absolute_error(target_test, target_predictions)
mse_value = mean_squared_error(target_test, target_predictions)
r2_value = r2_score(target_test, target_predictions)

print("\n Model Performance Metrics:")
print(f" Mean Absolute Error (MAE): {mae_value:.2f}")
print(f" Mean Squared Error (MSE): {mse_value:.2f}")
print(f" R-squared Score: {r2_value:.2f}")

# Step 8: Make Predictions
new_sample = np.array([[200, 40, 60]])  # Example: TV=200, Radio=40, Newspaper=60
new_sample_scaled = scaler_obj.transform(new_sample)
predicted_output = regression_model.predict(new_sample_scaled)

print("\n Predicted Sales Revenue:", predicted_output[0])
