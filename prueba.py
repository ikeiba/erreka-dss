# ----------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import String, Integer, Float, Boolean, DateTime
import os
from dotenv import load_dotenv

# ----------------------------------------------------------
# 2. DATABASE CONFIGURATION
# ----------------------------------------------------------
load_dotenv()

DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1") 
DB_USER = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("ROOT_PASSWORD")
DB_NAME = os.getenv("MYSQL_DATABASE", "erreka_dss_exam")
DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))

# ----------------------------------------------------------
# 3. ETL CONFIGURATION & METADATA
# ----------------------------------------------------------
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
    "installed_base": "datasets/installed_base.csv"
}

# Diccionario de metadatos: Tipos, PKs (solo donde aplique) y nulos permitidos
TABLE_SCHEMAS = {
    "alert_types": {
        "pk": "alert_type_id",
        "not_null": ["alert_type_id", "alert_type", "technical_severity"],
        "dtypes": {
            "alert_type_id": String(20),
            "alert_type": String(100),
            "description": String(255),
            "technical_severity": String(50),
            "potential_operational_impact": String(50),
            "safety_related": String(10)
        }
    },
    "context_criticality": {
        "pk": "context_criticality_id",
        "not_null": ["context_criticality_id", "customer_type", "environment_type"],
        "dtypes": {
            "context_criticality_id": String(20),
            "customer_type": String(50),
            "environment_type": String(100),
            "criticality_level": String(50),
            "sla_category": String(50)
        }
    },
    "door_type_usage_catalog": {
        "pk": "door_type, usage_scenario, installation_environment",
        "not_null": ["door_type", "usage_scenario", "installation_environment"],
        "dtypes": {
            "door_type": String(100),
            "usage_scenario": String(50),
            "installation_environment": String(100),
            "criticality_level": String(50),
            "operational_complexity": String(50),
            "maintenance_intensity": String(50),
            "estimated_cycles_day": Integer()
        }
    },
    "doors_registry": {
        "pk": "door_id",
        "not_null": ["door_id", "country_id", "door_type"],
        "dtypes": {
            "door_id": String(50),
            "country_id": String(10),
            "country_name": String(100),
            "door_type": String(100),
            "usage_scenario": String(50),
            "installation_environment": String(100),
            "customer_type": String(50),
            "context_criticality_id": String(20)
        }
    },
    "erreka_maintenance_history": {
        "pk": "door_id",
        "not_null": ["door_id", "last_maintenance_date", "maintenance_type"],
        "dtypes": {
            "door_id": String(50),
            "last_maintenance_date": DateTime(),
            "maintenance_type": String(50),
            "number_of_past_failures": Integer(),
            "days_since_last_failure": Integer(),
            "days_to_next_failure": Float(),
            "failed_next_30_days": String(10)
        }
    },
    "garage_operations_log": {
        # ELIMINADA LA PRIMARY KEY PARA ACEPTAR DUPLICADOS DE TELEMETRÍA
        "not_null": ["timestamp", "door_id"],
        "dtypes": {
            "timestamp": DateTime(),
            "door_id": String(50),
            "encoder_position": Integer(),
            "encoder_speed": Float(),
            "motor_torque": Float()
        }
    },
    "incident_events": {
        "pk": "incident_id",
        "not_null": ["incident_id", "timestamp", "door_id", "alert_type_id"],
        "dtypes": {
            "incident_id": String(50),
            "timestamp": DateTime(),
            "door_id": String(50),
            "country_id": String(10),
            "alert_type_id": String(20)
        }
    },
    "industrial_operations_log": {
        # ELIMINADA LA PRIMARY KEY PARA ACEPTAR DUPLICADOS DE TELEMETRÍA
        "not_null": ["timestamp", "door_id"],
        "dtypes": {
            "timestamp": DateTime(),
            "door_id": String(50),
            "encoder_position": Integer(),
            "encoder_speed": Float(),
            "cycle_counter": Integer(),
            "motor_current": Float(),
            "motor_torque": Float(),
            "vibration_level": Float(),
            "motor_temperature": Float()
        }
    },
    "installed_base": {
        "pk": "country_id, door_type, usage_scenario, installation_environment",
        "not_null": ["country_id", "door_type", "installed_doors"],
        "dtypes": {
            "country_id": String(10),
            "country_name": String(100),
            "door_type": String(100),
            "usage_scenario": String(50),
            "installation_environment": String(100),
            "installed_doors": Integer(),
            "customer_type": String(50)
        }
    },
    "pedestrian_operations_log": {
        # ELIMINADA LA PRIMARY KEY PARA ACEPTAR DUPLICADOS DE TELEMETRÍA
        "not_null": ["timestamp", "door_id"],
        "dtypes": {
            "timestamp": DateTime(),
            "door_id": String(50),
            "encoder_position": Integer(),
            "encoder_speed": Float(),
            "cycle_counter": Integer(),
            "motor_torque": Float(),
            "vibration_level": Float(),
            "motor_temperature": Float()
        }
    },
    "risk_factors": {
        "pk": "risk_factor_id",
        "not_null": ["risk_factor_id", "risk_factor"],
        "dtypes": {
            "risk_factor_id": String(20),
            "risk_factor": String(100),
            "description": String(255),
            "door_type": String(100),
            "usage_scenario": String(50),
            "risk_dimension": String(50)
        }
    }
}

# ----------------------------------------------------------
# 4. CREATE CONNECTION TO MYSQL
# ----------------------------------------------------------
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
insp = inspect(engine)

for table_name, file_path in ETL_FILES.items():
    print(f"\n---> Processing table: '{table_name}'")

    table_exist = insp.has_table(table_name)

    if table_exist:
        print(f"     [i] Table '{table_name}' already exists. Skipping...")
        continue
    
    if not os.path.exists(file_path):
        print(f"     [!] WARNING: File {file_path} not found. Skipping table...")
        continue

    schema_config = TABLE_SCHEMAS.get(table_name, {})

    try:
        # --- EXTRACT ---
        df = pd.read_csv(file_path)
        print(f"     [+] Extracted: {len(df)} rows from {file_path}")

        # --- TRANSFORM ---
        # 1. Eliminar filas totalmente vacías
        df.dropna(how="all", inplace=True)

        # 2. Limpieza de strings (AÑADIDO "str" PARA EVITAR EL WARNING DE PANDAS)
        for col in df.select_dtypes(include=["object", "str"]).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df.apply(lambda x: None if x[col] == 'nan' else x[col], axis=1)

        # 3. Parseo de fechas (general)
        for col in df.columns:
            if (('date' in col.lower() or 'time' in col.lower()) and 'downtime' not in col.lower()):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        
        # 4. Lógica de nulos obligatorios (NOT NULL)
        if "not_null" in schema_config:
            cols_to_check = [c for c in schema_config["not_null"] if c in df.columns]
            initial_len = len(df)
            df.dropna(subset=cols_to_check, inplace=True)
            dropped = initial_len - len(df)
            if dropped > 0:
                print(f"     [!] Dropped {dropped} rows due to NULL values in mandatory columns.")

        print("     [+] Transformed: Basic cleaning, formatting, and null checks completed.")

        # --- LOAD ---
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=10000,
            dtype=schema_config.get("dtypes", None)
        )
        print(f"     [+] Loaded: Data successfully inserted into MySQL table '{table_name}'.")

        # --- ADD PRIMARY KEY ---
        if "pk" in schema_config:
            with engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY ({schema_config['pk']});"))
                conn.commit()
            print(f"     [+] Schema: Primary Key ({schema_config['pk']}) applied to '{table_name}'.")

    except Exception as e:
        print(f"     [x] ERROR: Failed to process '{table_name}'. Reason: {e}")

# ----------------------------------------------------------
# 6. ETL FINISHED
# ----------------------------------------------------------
print("\n=====================================================")
print("ETL PROCESS COMPLETED SUCCESSFULLY.")
print("=====================================================")



# ----------------------------------------------------------
# 7. Validation
# ----------------------------------------------------------

# Row counts: compare CSV file shape vs database table shape
for table, file_path in ETL_FILES.items():
    if not os.path.exists(file_path):
        print(f"\n{table}.............")
        print(f"CSV file {file_path} not found. Skipping validation.")
        continue

    df = pd.read_csv(file_path)
    with engine.connect() as con:
        try:
            sql_df = pd.read_sql(text(f"SELECT * FROM {table}"), con)
            print(f"\n{table}.............")
            print(f"Shape tabla CSV ={df.shape}")
            print(f"Shape tabla SQL ={sql_df.shape}")

        except Exception as e:
            print(f"SQL Error Select in table: {table} --- Error: {e}")
            continue

# Traceability example query
sql_query = text("""
SELECT 
    log.timestamp AS log_timestamp,
    reg.door_id,
    reg.installation_environment,
    ctx.criticality_level,
    ctx.sla_category
FROM pedestrian_operations_log AS log
JOIN doors_registry AS reg ON log.door_id = reg.door_id
JOIN context_criticality AS ctx ON reg.installation_environment = ctx.environment_type
LIMIT 10;
""")

with engine.connect() as con:
    try:
        sql_result = pd.read_sql(sql_query, con)
        print(f"\nVALIDATION QUERY RESULT:.............")
        print(sql_result)

    except Exception as e:
        print(f"SQL Validation query Error: {e}")
        pass


# Criticality_level = impact if the door fails (safety of people, operational continuity, operating environment)