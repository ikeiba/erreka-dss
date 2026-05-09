import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# MySQL credentials
DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1") 
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("ROOT_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "erreka_dss_demo")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Read the csv with the predictions
RUTA_CSV_NUEVO = "datasets/erreka_maintenance_history_updated.csv" 
df_actualizado = pd.read_csv(RUTA_CSV_NUEVO)

# We take only door_id and the updated columns
df_update = df_actualizado[['door_id', 'failed_next_30_days', 'probability_score', 'risk_level']]

# Create the temporal table
print("Updating data to the temporal table (Staging)...")
df_update.to_sql("staging_maintenance", con=engine, if_exists="replace", index=False)

# Update the real table without breaking dependencies
print("Applying changes to the database...")

with engine.begin() as conn:
    try:
        conn.execute(text("ALTER TABLE erreka_maintenance_history ADD COLUMN probability_score FLOAT;"))
        conn.execute(text("ALTER TABLE erreka_maintenance_history ADD COLUMN risk_level VARCHAR(50);"))
        print("New columns added to the table")
    except Exception as e:
        print("Warning: The columns already exits (or have been a warning), continuing...")

    # Make the update crossing the tables
    query_update = text("""
        UPDATE erreka_maintenance_history main
        JOIN staging_maintenance staging ON main.door_id = staging.door_id
        SET main.failed_next_30_days = staging.failed_next_30_days,
            main.probability_score = staging.probability_score,
            main.risk_level = staging.risk_level;
    """)
    conn.execute(query_update)
    print("Raws updated with the created predictions")
    
    # Delete the temporary table
    conn.execute(text("DROP TABLE staging_maintenance;"))
    print("Cleaning of the temporary table completed")

print("Process completed")