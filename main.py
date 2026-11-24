
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sklearn

# print("Electrical Fault Detection Project")
# print("-----------------------------------")
# print("NumPy version:", np.__version__)
# print("Pandas version:", pd.__version__)
# print("Scikit-learn version:", sklearn.__version__)
# print("All libraries imported successfully!")


# #*********manual basic code********
# def detect_fault(voltage, current):

#     # Define reference limits (demonstration values)
#     normal_voltage = 230
#     min_voltage = 207
#     max_voltage = 253
#     max_current = 20

#     # Fault detection
#     if current > 30 and voltage < 180:
#         return "Short-Circuit-Like Condition"

#     elif voltage > max_voltage:
#         return "Overvoltage"

#     elif voltage < min_voltage:
#         return "Undervoltage"

#     elif current > max_current:
#         return "Overcurrent"

#     else:
#         return "Normal Condition"


# # Get electrical measurements
# voltage = float(input("Enter Voltage (V): "))
# current = float(input("Enter Current (A): "))

# # Detect fault
# result = detect_fault(voltage, current)

# # Display result
# print("\n----- Electrical Fault Detection -----")
# print("Voltage:", voltage, "V")
# print("Current:", current, "A")
# print("Detected Condition:", result)


# ******Read the CSV dataset using Python****
import pandas as pd

# Load the electrical fault dataset
data = pd.read_csv("dataset/electrical_data.csv")

# Display the dataset
print("Electrical Fault Detection Dataset")
print("----------------------------------")
print(data)

# Display the number of records
print("\nTotal records:", len(data))

# Understand the dataset

print("\nDataset Information:")
print(data.info())


print("\nFault Type Counts:")
print(data["fault_type"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())

# Prepare data for Machine Learning

from sklearn.model_selection import train_test_split

# Input features
X = data[["voltage", "current"]]

# Output labels
y = data["fault_type"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

print("\nMachine Learning Data Preparation")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# Train the Random Forest Model

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nRandom Forest Model Training")
print("Model trained successfully!")

# Test the model

from sklearn.metrics import accuracy_score

# Predict faults using testing data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Testing Results")
print("---------------------")
print("Actual Faults:", list(y_test))
print("Predicted Faults:", list(y_pred))
print("Model Accuracy:", accuracy * 100, "%")

# Detailed Model Evaluation

from sklearn.metrics import classification_report

print("\nClassification Report")
print("---------------------")

print(classification_report(y_test, y_pred, zero_division=0))

# Predict fault for new electrical readings

print("\nEnter Electrical Readings")

voltage = float(input("Enter Voltage (V): "))
current = float(input("Enter Current (A): "))

new_data = pd.DataFrame({
    "voltage": [voltage],
    "current": [current]
})

prediction = model.predict(new_data)

print("\nFault Detection Result")
print("Voltage:", voltage, "V")
print("Current:", current, "A")
print("Predicted Fault:", prediction[0])


# Electrical Fault Data Visualization

import matplotlib.pyplot as plt

# Create voltage vs current graph
for fault in data["fault_type"].unique():

    fault_data = data[data["fault_type"] == fault]

    plt.scatter(
        fault_data["voltage"],
        fault_data["current"],
        label=fault,
        s=100
    )

# Add graph details
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Electrical Fault Detection - Voltage vs Current")

plt.legend()
plt.grid(True)

# Save the graph
plt.savefig("graphs/fault_visualization.png")

# Display the graph
plt.show()

# Fault Category Distribution Graph

fault_counts = data["fault_type"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(fault_counts.index, fault_counts.values)

plt.xlabel("Fault Category")
plt.ylabel("Number of Records")
plt.title("Electrical Fault Category Distribution")

plt.grid(axis="y")

plt.savefig("graphs/fault_distribution.png")

plt.show()

# Confusion Matrix Visualization

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Electrical Fault Detection - Confusion Matrix")

# Save graph
plt.savefig("graphs/confusion_matrix.png")

plt.show()