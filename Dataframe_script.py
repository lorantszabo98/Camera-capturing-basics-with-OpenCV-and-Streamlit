import pandas as pd

# Sample data
data = {
    "Name": ["John", "Alice", "Bob", "Kinga", "Lóránt"],
    "Date": ["2023-11-10", "2023-11-10", "2023-11-10", "2023-11-09", "2023-11-08"],
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Age": ["8-12", "25-32", "25-32", "8-12", "0-8"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("pages/data/data.csv", index=False)
