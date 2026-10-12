import os
import pandas as pd
import numpy as np

output_filename = "FULL_DATATHON_PACK.txt"
IGNORE_DIRS = {'.venv', '.dist', '.git', '__pycache__', 'COLAB_REQUIRED_FILES'}

with open(output_filename, "w", encoding="utf-8") as out:
    out.write("===================================================\n")
    out.write("     REACT DATATHON - COMPLETE PROJECT DOSSIER     \n")
    out.write("===================================================\n\n")

    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file in [output_filename, "bundle.py"]:
                continue
            
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, ".")
            out.write(f"\n\n{'='*80}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{'='*80}\n\n")

            try:
                # Code, Markdown, Text, Logs
                if file.endswith(('.md', '.py', '.txt', '.json', '.log')):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        out.write(f.read())

                # CSV Datasets
                elif file.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    out.write(f"--- CSV Summary ({file}) ---\n")
                    out.write(f"Shape: {df.shape}\n")
                    out.write("Columns & Dtypes:\n" + str(df.dtypes) + "\n\n")
                    out.write("Missing Values:\n" + str(df.isnull().sum()) + "\n\n")
                    out.write("Head (3 rows):\n" + df.head(3).to_string() + "\n")

                # Parquet Files
                elif file.endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                    out.write(f"--- Parquet Summary ({file}) ---\n")
                    out.write(f"Shape: {df.shape}\n")
                    out.write("Columns & Dtypes:\n" + str(df.dtypes) + "\n\n")
                    out.write("Head (3 rows):\n" + df.head(3).to_string() + "\n")

                # Numpy Files (.npy)
                elif file.endswith('.npy'):
                    arr = np.load(filepath)
                    out.write(f"--- NPY Array Summary ({file}) ---\n")
                    out.write(f"Shape: {arr.shape}, Dtype: {arr.dtype}\n")

            except Exception as e:
                out.write(f"[Could not parse file content: {e}]\n")

print(f"Done! Everything merged into: {output_filename}")