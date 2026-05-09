# ----------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# ----------------------------------------------------------
# 2. DATABASE CONFIGURATION
# ----------------------------------------------------------
load_dotenv()

DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1") 
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("ROOT_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "erreka_dss")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))

# ----------------------------------------------------------
# 3. ETL CONFIGURATION
# ----------------------------------------------------------
# Define the new table name and the path to the predicted dataset
TABLE_NAME = "predicted_maintenance_history"
FILE_PATH = "regression/predicted_maintenance_history.csv" # Adjust to "Datasets/..." if you moved the file there

# ----------------------------------------------------------
# 4. CREATE CONNECTION TO MYSQL
# ----------------------------------------------------------
engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Connection to MySQL established successfully.\n")
print("=====================================================")
print("STARTING ETL PROCESS FOR PREDICTIONS")
print("=====================================================")

# ----------------------------------------------------------
# 5. EXTRACT, TRANSFORM, LOAD
# ----------------------------------------------------------
print(f"\n---> Processing table: '{TABLE_NAME}'")

if not os.path.exists(FILE_PATH):
    print(f"[!] ERROR: File {FILE_PATH} not found. Please run the regression pipeline first.")
else:
    try:
        # --- EXTRACT ---
        df = pd.read_csv(FILE_PATH)
        print(f"[+] Extracted: {len(df)} rows from {FILE_PATH}")

        # --- TRANSFORM ---
        df.dropna(how="all", inplace=True)

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', None)

        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        print("     [+] Transformed: Basic cleaning and formatting completed.")

        # --- LOAD ---
        # Note: We use if_exists="replace" here. 
        # Since this is a predictions table, every time you run the model you get updated 
        # predictions for all doors, so you want to overwrite the old predictions table.
        df.to_sql(
            name=TABLE_NAME,
            con=engine,
            if_exists="replace", 
            index=False,
            chunksize=10000
        )
        print(f"     [+] Loaded: Data successfully inserted/replaced into MySQL table '{TABLE_NAME}'.")

    except Exception as e:
        print(f"     [x] ERROR: Failed to process '{TABLE_NAME}'. Reason: {e}")

# ----------------------------------------------------------
# 6. END
# ----------------------------------------------------------
print("\n=====================================================")
print("PREDICTIONS ETL PROCESS COMPLETED.")
print("=====================================================")