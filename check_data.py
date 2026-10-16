
import pandas as pd

# Load the training dataset
file_path = "messy-food-waste-prediction-dataset/train.csv"
data = pd.read_csv(file_path)

# Display the dataset information
print("\n--- First 5 Rows ---")
print(data.head())

print("\n--- Column Names ---")
print(data.columns.tolist())

print("\n--- Dataset Size ---")
print("Rows and columns:", data.shape)

print("\n--- Missing Values ---")
print(data.isnull().sum())
