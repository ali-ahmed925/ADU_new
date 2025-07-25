import os
import pandas as pd
from itertools import product

# Set weights and base path
domain_weights = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0 ]
mmd_weights = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
base_path = "/nas/data/kawamura/ADU"

# Initialize tables for F values from each Forgetdomain
tables = {
    "Forgetdomain1": pd.DataFrame(index=domain_weights, columns=mmd_weights),
    "Forgetdomain2": pd.DataFrame(index=domain_weights, columns=mmd_weights),
    "Forgetdomain3": pd.DataFrame(index=domain_weights, columns=mmd_weights),
}

for dw, mw in product(domain_weights, mmd_weights):
    file_path = f"{base_path}/domain_weight_{dw}/mmd_weight_{mw}/BBF/results_seed1_datasetseed0.csv"
    if not os.path.exists(file_path):
        print(f"Missing: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path, header=None)
        # Row 2 (index 1) contains the labels, Row 3 (index 2) contains the values
        values = df.iloc[2].dropna().tolist()  # row with actual data

        # Extract F values (index: H, A, F → step of 3, starting from 2)
        f_values = {
            "Forgetdomain1": float(values[2]),  # H,A,F -> index 4 is F
            "Forgetdomain2": float(values[5]),
            "Forgetdomain3": float(values[8]),
        }

        for domain, f in f_values.items():
            tables[domain].at[dw, mw] = f

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save tables
for domain, df in tables.items():
    df = df.astype(float)  # convert strings to float if necessary
    df.to_csv(f"summary_F_{domain}.csv")
    print(f"Saved summary_F_{domain}.csv")
