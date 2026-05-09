"""
Erreka - Predictive Maintenance Pipeline (No Data Leakage)
----------------------------------------------------------
This script builds a robust regression model to predict 'days_to_next_failure'.
It specifically prevents Temporal Data Leakage by calculating operational features 
(temperature, torque, etc.) strictly in the time window BEFORE a given failure.

Output: 
- Evaluates the model on historical data.
- Generates a ranked list of doors currently active, sorted by urgency.
- Saves the final modeling dataset to 'regression/regression_failure.csv'.
- Saves predictions to 'regression/predicted_maintenance_history.csv' 
- DIRECTLY updates the 'predicted_maintenance_history' table in MySQL.
"""

import os
import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Scikit-Learn tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore') # Suppress pandas chained assignment warnings for cleaner output

# ----------------------------------------------------------
# DATABASE CONFIGURATION
# ----------------------------------------------------------
def get_db_engine():
    """Carga las credenciales del .env y crea la conexión a MySQL."""
    load_dotenv()
    DB_HOST = os.getenv("MYSQL_HOST", "127.0.0.1") 
    DB_USER = os.getenv("MYSQL_USER", "root")
    DB_PASSWORD = os.getenv("ROOT_PASSWORD")
    DB_NAME = os.getenv("MYSQL_DATABASE", "erreka_dss_demo")
    DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))

    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return engine

# --- SECTION 1: DATA LOADING ---

def load_data(engine):
    print("Loading datasets from MySQL database...")
    
    incidents_df = pd.read_sql_table("incident_events", con=engine)
    maintenance_df = pd.read_sql_table("erreka_maintenance_history", con=engine)
    
    ped_logs = pd.read_sql_table("pedestrian_operations_log", con=engine)
    gar_logs = pd.read_sql_table("garage_operations_log", con=engine)
    ind_logs = pd.read_sql_table("industrial_operations_log", con=engine)
    
    all_logs_df = pd.concat([ped_logs, gar_logs, ind_logs], ignore_index=True)
    
    # FIX: forzar str puro en TODOS los dataframes que vienen de SQL
    for d in (incidents_df, maintenance_df, all_logs_df):
        d.columns = [str(c) for c in d.columns]
    
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    all_logs_df['timestamp'] = pd.to_datetime(all_logs_df['timestamp'])
    
    return incidents_df, maintenance_df, all_logs_df

# --- SECTION 2: TARGET & TIME-WINDOW GENERATION ---

def build_target_dataframe(incidents_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    print("Building target variables and time windows...")
    
    df = incidents_df.sort_values(["door_id", "timestamp"]).copy()
    df["next_timestamp"] = df.groupby("door_id")["timestamp"].shift(-1)
    df["prev_timestamp"] = df.groupby("door_id")["timestamp"].shift(1)
    df["days_to_next_failure"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds() / (60 * 60 * 24)
    
    all_doors = set(maintenance_df['door_id'])
    incident_doors = set(df['door_id'])
    missing_doors = list(all_doors - incident_doors)
    
    if missing_doors:
        current_time = df['timestamp'].max()
        missing_rows = []
        for door in missing_doors:
            static_info = maintenance_df[maintenance_df['door_id'] == door].iloc[0]
            door_type = "ECOline Home" if door.startswith("G-") else ("ROLLfast Industrial 300" if door.startswith("I-") else "ECOline Pedestrian")
            
            missing_rows.append({
                'door_id': door,
                'timestamp': current_time, 
                'next_timestamp': pd.NaT,
                'prev_timestamp': pd.NaT, 
                'days_to_next_failure': np.nan, 
                'door_type': door_type,
                'usage_scenario': 'low', 
                'installation_environment': 'Unknown'
            })
            
        missing_df = pd.DataFrame(missing_rows)
        df = pd.concat([df, missing_df], ignore_index=True)
        print(f"Added {len(missing_doors)} active doors with zero previous incidents for prediction.")
        
    return df

# --- SECTION 3: TEMPORAL FEATURE ENGINEERING ---

def extract_windowed_features(target_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting time-windowed features (this may take a moment)...")
    
    logs_df = logs_df.sort_values(["door_id", "timestamp"])
    features_list = []
    
    for _, row in target_df.iterrows():
        door = row['door_id']
        t_curr = row['timestamp']
        t_prev = row['prev_timestamp']
        
        door_logs = logs_df[logs_df['door_id'] == door]
        mask = (door_logs['timestamp'] <= t_curr)
        
        if pd.notnull(t_prev):
            mask = mask & (door_logs['timestamp'] > t_prev)
            
        window_logs = door_logs[mask]
        
        row_features = {
            'incident_id': row.get('incident_id', f"{door}_{t_curr}"),
            'door_id': door,
            'timestamp': t_curr,
            'days_to_next_failure': row['days_to_next_failure'],
            'door_type': row['door_type'],
            'usage_scenario': row['usage_scenario'],
            'installation_environment': row.get('installation_environment', 'Unknown')
        }
        
        if not window_logs.empty:
            row_features['motor_temp_mean'] = window_logs['motor_temperature'].mean() if 'motor_temperature' in window_logs else np.nan
            row_features['motor_temp_max'] = window_logs['motor_temperature'].max() if 'motor_temperature' in window_logs else np.nan
            row_features['motor_torque_mean'] = window_logs['motor_torque'].mean() if 'motor_torque' in window_logs else np.nan
            row_features['motor_torque_max'] = window_logs['motor_torque'].max() if 'motor_torque' in window_logs else np.nan
            
            if 'emergency_stop_activated' in window_logs:
                row_features['emergency_stops_count'] = window_logs['emergency_stop_activated'].fillna(False).astype(bool).sum()
            else:
                row_features['emergency_stops_count'] = 0
                
            if 'safety_photocell_blocked' in window_logs:
                row_features['photocell_blocks_count'] = window_logs['safety_photocell_blocked'].fillna(False).astype(bool).sum()
            else:
                row_features['photocell_blocks_count'] = 0
        else:
            row_features['motor_temp_mean'] = np.nan
            row_features['motor_temp_max'] = np.nan
            row_features['motor_torque_mean'] = np.nan
            row_features['motor_torque_max'] = np.nan
            row_features['emergency_stops_count'] = 0
            row_features['photocell_blocks_count'] = 0
            
        features_list.append(row_features)
        
    return pd.DataFrame(features_list)

# --- SECTION 4: DATASET INTEGRATION ---

# --- SECTION 4: DATASET INTEGRATION ---

def build_modeling_dataset(temporal_features_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    print("Merging temporal and static features...")
    static_df = maintenance_df[['door_id', 'maintenance_type', 'number_of_past_failures']].copy()
    final_df = pd.merge(temporal_features_df, static_df, on='door_id', how='left')
    
    # SOLUCIÓN: Convertir todos los nombres de columnas a string normal
    # para evitar conflictos entre SQLAlchemy 'quoted_name' y los 'str' de Pandas
    final_df.columns = [str(c) for c in final_df.columns]
    
    return final_df

# --- SECTION 5: MODEL TRAINING & INFERENCE ---

def train_evaluate_and_rank(df: pd.DataFrame, original_maintenance_df: pd.DataFrame):
    print("\n--- Training Predictive Model ---")

    df.columns = [str(c) for c in df.columns]
    
    numeric_features = [
        'motor_temp_mean', 'motor_temp_max', 'motor_torque_mean', 
        'motor_torque_max', 'emergency_stops_count', 'photocell_blocks_count',
        'number_of_past_failures'
    ]
    categorical_features = ['door_type', 'usage_scenario', 'installation_environment', 'maintenance_type']
    
    historical_data = df[df['days_to_next_failure'].notna()].copy()
    active_doors = df[df['days_to_next_failure'].isna()].copy()
    
    X_hist = historical_data[numeric_features + categorical_features]
    y_hist = historical_data['days_to_next_failure']
    X_active = active_doors[numeric_features + categorical_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.30, random_state=42)
    
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== Regression Evaluation on Test Set ===")
    print(f"MAE  (days): {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE (days): {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"R^2        : {r2_score(y_test, y_pred):.3f}")
    
    print("\n--- Generating Decision Support Output ---")
    active_doors['predicted_days_to_failure'] = model.predict(X_active)
    ranking = active_doors.sort_values("predicted_days_to_failure", ascending=True)
    
    cols_to_show = ['door_id', 'door_type', 'maintenance_type', 'predicted_days_to_failure']
    print(f"\n=== Top 10 Doors Predicted to Fail Sooner (Urgent Maintenance) ===")
    print(ranking[cols_to_show].head(10).to_string(index=False))
    
    # Preparamos el dataframe final con las predicciones unidas
    maintenance_update_df = original_maintenance_df.copy()
    prediction_map = dict(zip(active_doors['door_id'], active_doors['predicted_days_to_failure']))
    maintenance_update_df['days_to_next_failure'] = maintenance_update_df['door_id'].map(prediction_map)
    
    return maintenance_update_df

# --- MAIN EXECUTION ---

def main():
    # 1. Crear conexión a la base de datos
    engine = get_db_engine()
    
    # 2. Leer datos directamente de MySQL
    incidents_df, maintenance_df, all_logs_df = load_data(engine)
    
    # 3. Construir targets y features temporales
    target_df = build_target_dataframe(incidents_df, maintenance_df)
    temporal_features_df = extract_windowed_features(target_df, all_logs_df)
    
    # 4. Construir dataset final para el modelo
    final_modeling_df = build_modeling_dataset(temporal_features_df, maintenance_df)
    
    # Guardar CSV intermedio (opcional pero útil para debug)
    modeling_output_path = "regression/regression_failure.csv"
    final_modeling_df.to_csv(modeling_output_path, index=False)
    print(f"\nModeling dataset successfully saved to: {modeling_output_path}")
    
    # 5. Entrenar modelo y obtener el dataframe final con las predicciones añadidas
    predicted_df = train_evaluate_and_rank(final_modeling_df, maintenance_df)
    
    # 6. GUARDAR EN CSV LOCAL (Mantiene la ruta que pediste)
    predictions_output_path = "regression/predicted_maintenance_history.csv"
    predicted_df.to_csv(predictions_output_path, index=False)
    print(f"\nPredictions successfully saved to local CSV: {predictions_output_path}")

    # 7. CARGA DIRECTA A MYSQL (Sustituye al script load_predictions_etl.py)
    print("\n=====================================================")
    print("STARTING DIRECT LOAD TO MYSQL")
    print("=====================================================")
    
    # Limpieza básica para asegurar compatibilidad en SQL (igual que en tu ETL)
    predicted_df.dropna(how="all", inplace=True)
    
    for col in predicted_df.select_dtypes(include=["object"]).columns:
        predicted_df[col] = predicted_df[col].astype(str).str.strip()
        predicted_df[col] = predicted_df[col].replace('nan', None)

    # Lógica de fechas restaurada
    for col in predicted_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                predicted_df[col] = pd.to_datetime(predicted_df[col])
            except Exception:
                pass

    try:
        predicted_df.to_sql(
            name="predicted_maintenance_history",
            con=engine,
            if_exists="replace", # Reemplaza los datos con la nueva predicción de hoy
            index=False,
            chunksize=10000
        )
        print("     [+] Loaded: Data successfully inserted/replaced into MySQL table 'predicted_maintenance_history'.")
    except Exception as e:
        print(f"     [x] ERROR: Failed to upload to MySQL. Reason: {e}")
        
    print("\n=====================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY.")
    print("=====================================================")

if __name__ == "__main__":
    main()