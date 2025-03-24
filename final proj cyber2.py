import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Ignore warnings
wr.filterwarnings("ignore")

# loading and reading dataset
data = pd.read_csv("final proj cyber seq.csv")
print(data.head())

# Check basic information about the dataset
print(data.info())

# Define number of records
num_records = 100
print(f"Number of records: {num_records}")

# Check for missing values
print(data.isnull().sum())

# Ensure "Failed_Login_Count" exists
if "Failed_Login_Count" not in data.columns:
    data["Failed_Login_Count"] = data.groupby("User_ID")["Login_Status"].transform(lambda x: (x == "Failed").sum())

# Convert timestamp to datetime
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["Hour"] = data["Timestamp"].dt.hour
data["Date"] = data["Timestamp"].dt.date

# Filter failed logins
failed_logins = data[data["Login_Status"] == "Failed"]

# Count failed logins per location
failed_counts = failed_logins.groupby("Location").size().reset_index(name="Failed_Logins")

## Bar Chart: Failed Logins by Location
plt.figure(figsize=(8, 4))
sns.barplot(x="Failed_Logins", y="Location", data=failed_counts, dodge=False, palette="Reds")
plt.title("Failed Logins by Location")
plt.xlabel("Number of Failed Logins")
plt.ylabel("Location")
plt.show()

## Heatmap: Login Activity per Hour
time_heatmap_data = data.pivot_table(index="Hour", columns="Date", values="User_ID", aggfunc="count")
plt.figure(figsize=(12, 6))
sns.heatmap(time_heatmap_data, cmap="coolwarm", linewidths=0.5)
plt.title("Login Activity Heatmap (Hourly)")
plt.xlabel("Date")
plt.ylabel("Hour of the Day")
plt.show()

# Linear Regression: Predicting Failed Logins
## Prepare Data
df_filtered = data[["Access_Duration", "Failed_Login_Count"]].dropna()

if df_filtered.empty:
    print("Error: No data available for Linear Regression.")
else:
    X = df_filtered[["Access_Duration"]]
    y = df_filtered["Failed_Login_Count"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict failed logins
    df_filtered["Predicted_Failed_Logins"] = model.predict(X)

    # Plot Actual vs. Predicted Failed Logins
    plt.figure(figsize=(10, 5))
    plt.scatter(df_filtered["Access_Duration"], df_filtered["Failed_Login_Count"], label="Actual Data", alpha=0.5)
    plt.plot(df_filtered["Access_Duration"], df_filtered["Predicted_Failed_Logins"], color="red", label="Regression Line")
    plt.xlabel("Access Duration (seconds)")
    plt.ylabel("Failed Login Count")
    plt.title("Linear Regression: Predicting Failed Logins")
    plt.legend()
    plt.show()
