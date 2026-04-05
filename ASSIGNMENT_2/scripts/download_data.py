"""
Script to download the HR_comma_sep.csv dataset
"""
import os
import urllib.request

# URL from a reliable public source
url = "https://raw.githubusercontent.com/sumit2405/HR-Employee-Attrition/main/HR_comma_sep.csv"
data_dir = os.path.join("ASSIGNMENT_2", "data")
os.makedirs(data_dir, exist_ok=True)

output_path = os.path.join(data_dir, "HR_comma_sep.csv")

if not os.path.exists(output_path):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Dataset saved to {output_path}")
else:
    print(f"Dataset already exists at {output_path}")
