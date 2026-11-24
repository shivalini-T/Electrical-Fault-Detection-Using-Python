

import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/electrical_data.csv")

# Prepare training data
X = data[["voltage", "current"]]
y = data["fault_type"]

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)


# Fault detection function
def detect_fault():
    try:
        voltage = float(voltage_entry.get())
        current = float(current_entry.get())

        new_data = pd.DataFrame({
            "voltage": [voltage],
            "current": [current]
        })

        prediction = model.predict(new_data)[0]

        result_label.config(
            text="⚡ " + prediction.upper(),
            fg="#00e676"
        )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numerical values."
        )


# Main window
root = tk.Tk()
root.title("Electrical Fault Detection System")
root.geometry("600x600")
root.configure(bg="#101c36")
root.resizable(False, False)

# Main heading
heading = tk.Label(
    root,
    text="ELECTRICAL FAULT DETECTION",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#101c36"
)
heading.pack(pady=25)

# Subtitle
subtitle = tk.Label(
    root,
    text="AI-Based Electrical Monitoring System",
    font=("Arial", 11),
    fg="#a9c5e8",
    bg="#101c36"
)
subtitle.pack(pady=5)

# Input section
input_frame = tk.Frame(root, bg="#1c2d50", padx=35, pady=25)
input_frame.pack(pady=25)

# Voltage input
tk.Label(
    input_frame,
    text="Voltage (V)",
    font=("Arial", 12, "bold"),
    fg="white",
    bg="#1c2d50"
).pack()

voltage_entry = tk.Entry(
    input_frame,
    font=("Arial", 14),
    width=25,
    justify="center"
)
voltage_entry.pack(pady=10)

# Current input
tk.Label(
    input_frame,
    text="Current (A)",
    font=("Arial", 12, "bold"),
    fg="white",
    bg="#1c2d50"
).pack()

current_entry = tk.Entry(
    input_frame,
    font=("Arial", 14),
    width=25,
    justify="center"
)
current_entry.pack(pady=10)

# Detection button
detect_button = tk.Button(
    root,
    text="DETECT FAULT",
    command=detect_fault,
    font=("Arial", 13, "bold"),
    bg="#168aad",
    fg="white",
    activebackground="#34a0c4",
    activeforeground="white",
    width=20,
    height=2,
    cursor="hand2"
)
detect_button.pack(pady=10)

# Result heading
tk.Label(
    root,
    text="DETECTION RESULT",
    font=("Arial", 11, "bold"),
    fg="#a9c5e8",
    bg="#101c36"
).pack(pady=5)

# Result display
result_label = tk.Label(
    root,
    text="Waiting for input...",
    font=("Arial", 18, "bold"),
    fg="#f1c40f",
    bg="#1c2d50",
    width=30,
    pady=15
)
result_label.pack(pady=10)

# Footer
tk.Label(
    root,
    text="Educational Prototype | Python Machine Learning",
    font=("Arial", 9),
    fg="#8297b5",
    bg="#101c36"
).pack(side="bottom", pady=12)

# Run application
root.mainloop()
