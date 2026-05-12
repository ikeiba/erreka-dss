# ----------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------
import pandas as pd # We will use the easiest/most modern approach, which is pandas
from sqlalchemy import create_engine, inspect
import os
from dotenv import load_dotenv # This will allow us to "hide" our credentials

# ----------------------------------------------------------
# 2. DATABASE CONFIGURATION
# ----------------------------------------------------------
# We use the dotenv so that we don't have to write our private credentials in the script itself
load_dotenv()

# We could load all this variables from the .env, but we will just load the password as it is the only private credential
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1") 
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("ROOT_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "erreka_dss_demo")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))

# ----------------------------------------------------------
# 3. ETL CONFIGURATION
# ----------------------------------------------------------
# We create a dictionary where the key is the name of the table and the value is the route to the CSV
ETL_FILES = {
    "alert_types": "datasets/alert_types.csv",
    "context_criticality": "datasets/context_criticality.csv",
    "door_type_usage_catalog": "datasets/door_type_usage_catalog.csv",
    "doors_registry": "datasets/doors_registry.csv",
    "erreka_maintenance_history": "datasets/erreka_maintenance_history.csv",
    "garage_operations_log": "datasets/garage_operations_log.csv",
    "incident_events": "datasets/incident_events.csv",
    "industrial_operations_log": "datasets/industrial_operations_log.csv",
    "pedestrian_operations_log": "datasets/pedestrian_operations_log.csv",
    "risk_factors": "datasets/risk_factors.csv",
    "installed_base": "datasets/installed_base.csv",
}

# ----------------------------------------------------------
# 4. CREATE CONNECTION TO MYSQL
# ----------------------------------------------------------
# We create the engine to connect with our specific databaset, using our username and password
engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Connection to MySQL established successfully.\n")
print("=====================================================")
print("STARTING ETL PROCESS")
print("=====================================================")

# ----------------------------------------------------------
# 5. AUTOMATED ETL LOOP (EXTRACT, TRANSFORM, LOAD)
# ----------------------------------------------------------

# This will enable us to check if a table is already created
insp = inspect(engine)

# We iterate all the files in  a single loop
for table_name, file_path in ETL_FILES.items():
    print(f"\n---> Processing table: '{table_name}'")

    table_exist = insp.has_table(table_name)


    if table_exist:
        print(f"Table {table_name} already exist")
        continue
    
    # We first check if the file we are trying to load exists
    if not os.path.exists(file_path):
        print(f"[!] WARNING: File {file_path} not found. Skipping table...")
        continue

    try:
        # --- EXTRACT ---
        df = pd.read_csv(file_path)
        print(f"[+] Extracted: {len(df)} rows from {file_path}")

        # --- TRANSFORM ---
        # We remove any totally empty line (in case there is)
        df.dropna(how="all", inplace=True)

        # 2. We remove the blank spaces that sometimes strings have
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', None) # With this we ensure that if there was any empty space, we leave it like a None and not like a 'nan' string

        # 3. We parse date:
        # If the column has the word date or time, we transform it into the date format
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]) # Parsing the date type
                except Exception:
                    pass

        print("     [+] Transformed: Basic cleaning and formatting completed.")

        # --- LOAD ---
        # We store the csv in each own table of the database
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",  # 'replace' overwrites, we use 'append' if we want to add more rows
            index=False,
            chunksize=10000  # Sends data in batches of 10,000 (if we don't put this the process will fail with the biggest tables)
        )
        print(f"     [+] Loaded: Data successfully inserted into MySQL table '{table_name}'.")

    except Exception as e:
        print(f"     [x] ERROR: Failed to process '{table_name}'. Reason: {e}")

# ----------------------------------------------------------
# 6. END
# ----------------------------------------------------------
print("\n=====================================================")
print("ETL PROCESS COMPLETED SUCCESSFULLY.")
print("=====================================================")