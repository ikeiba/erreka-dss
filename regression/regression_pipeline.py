"""
Erreka - Predictive Maintenance Pipeline (No Data Leakage)
----------------------------------------------------------
This script builds a robust regression model to predict 'days_to_next_failure'.
It specifically prevents Temporal Data Leakage by calculating operational features 
(temperature, torque, etc.) strictly in the time window BEFORE a given failure.

Output: 
- Evaluates the model on historical data.
- Generates a ranked list of doors currently active, sorted by urgency.
- Saves the final modeling dataset to 'regression_failure.csv'.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Scikit-Learn tools
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore') # Suppress pandas chained assignment warnings for cleaner output

# --- SECTION 1: DATA LOADING ---

def load_data(data_dir: str = "../datasets"):
    """Loads all necessary CSV files from the specified directory."""
    path = Path(data_dir)
    
    print("Loading datasets...")
    incidents_df = pd.read_csv(path / "incident_events.csv")
    maintenance_df = pd.read_csv(path / "erreka_maintenance_history.csv")
    
    # Load and concatenate all operational logs into a single dataframe
    ped_logs = pd.read_csv(path / "pedestrian_operations_log.csv")
    gar_logs = pd.read_csv(path / "garage_operations_log.csv")
    ind_logs = pd.read_csv(path / "industrial_operations_log.csv")
    
    all_logs_df = pd.concat([ped_logs, gar_logs, ind_logs], ignore_index=True)
    
    # Ensure timestamps are datetime objects
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    all_logs_df['timestamp'] = pd.to_datetime(all_logs_df['timestamp'])
    
    return incidents_df, maintenance_df, all_logs_df

# --- SECTION 2: TARGET & TIME-WINDOW GENERATION ---

def build_target_dataframe(incidents_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts incidents chronologically per door and calculates the time until 
    the NEXT failure. Also captures the PREVIOUS failure timestamp to define 
    the feature extraction window (t_i-1, t_i].
    Includes 'phantom' rows for doors with zero incidents to enable inference.
    """
    print("Building target variables and time windows...")
    
    # 1. Process doors with incidents
    df = incidents_df.sort_values(["door_id", "timestamp"]).copy()
    df["next_timestamp"] = df.groupby("door_id")["timestamp"].shift(-1)
    df["prev_timestamp"] = df.groupby("door_id")["timestamp"].shift(1)
    df["days_to_next_failure"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds() / (60 * 60 * 24)
    
    # 2. Handle doors with NO incidents (The 'invincible' doors)
    # Find doors in maintenance that are NOT in incidents
    all_doors = set(maintenance_df['door_id'])
    incident_doors = set(df['door_id'])
    missing_doors = list(all_doors - incident_doors)
    
    if missing_doors:
        # We need a reference 'current time' for these doors to extract their logs up to now.
        # We'll use the maximum timestamp found in the entire incidents dataset as "now".
        current_time = df['timestamp'].max()
        
        missing_rows = []
        for door in missing_doors:
            # We fetch its static info from maintenance to match the columns
            static_info = maintenance_df[maintenance_df['door_id'] == door].iloc[0]
            # Assume it's a home door by default if not specified elsewhere (adjust if needed based on ID prefix)
            door_type = "ECOline Home" if door.startswith("G-") else ("ROLLfast Industrial 300" if door.startswith("I-") else "ECOline Pedestrian")
            
            missing_rows.append({
                'door_id': door,
                'timestamp': current_time, # Extract all logs up to this point
                'next_timestamp': pd.NaT,
                'prev_timestamp': pd.NaT, # No previous failure
                'days_to_next_failure': np.nan, # Needs prediction
                'door_type': door_type,
                'usage_scenario': 'low', # Safe default for missing contextual data
                'installation_environment': 'Unknown'
            })
            
        missing_df = pd.DataFrame(missing_rows)
        # Append the missing doors as active (NaN target) to the main target dataframe
        df = pd.concat([df, missing_df], ignore_index=True)
        print(f"Added {len(missing_doors)} active doors with zero previous incidents for prediction.")
        
    return df

# --- SECTION 3: TEMPORAL FEATURE ENGINEERING ---

def extract_windowed_features(target_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through each incident and calculates features based ONLY on the logs
    that occurred between the previous incident and the current incident.
    This strictly prevents temporal data leakage.
    """
    print("Extracting time-windowed features (this may take a moment)...")
    
    # Sort logs to optimize filtering
    logs_df = logs_df.sort_values(["door_id", "timestamp"])
    
    features_list = []
    
    for _, row in target_df.iterrows():
        door = row['door_id']
        t_curr = row['timestamp']
        t_prev = row['prev_timestamp']
        
        # Filter logs for this specific door
        door_logs = logs_df[logs_df['door_id'] == door]
        
        # Temporal filtering: Logs must be BEFORE or AT current incident
        mask = (door_logs['timestamp'] <= t_curr)
        
        # If there was a previous incident, logs must be strictly AFTER it
        if pd.notnull(t_prev):
            mask = mask & (door_logs['timestamp'] > t_prev)
            
        window_logs = door_logs[mask]
        
        # Initialize feature dictionary for this row
        row_features = {
            'incident_id': row.get('incident_id', f"{door}_{t_curr}"),
            'door_id': door,
            'timestamp': t_curr,
            'days_to_next_failure': row['days_to_next_failure'],
            'door_type': row['door_type'],
            'usage_scenario': row['usage_scenario'],
            'installation_environment': row.get('installation_environment', 'Unknown')
        }
        
        # Aggregate logic
        if not window_logs.empty:
            row_features['motor_temp_mean'] = window_logs['motor_temperature'].mean() if 'motor_temperature' in window_logs else np.nan
            row_features['motor_temp_max'] = window_logs['motor_temperature'].max() if 'motor_temperature' in window_logs else np.nan
            
            row_features['motor_torque_mean'] = window_logs['motor_torque'].mean() if 'motor_torque' in window_logs else np.nan
            row_features['motor_torque_max'] = window_logs['motor_torque'].max() if 'motor_torque' in window_logs else np.nan
            
            # Categorical counts (convert to bool, fillna with False, then sum)
            if 'emergency_stop_activated' in window_logs:
                row_features['emergency_stops_count'] = window_logs['emergency_stop_activated'].fillna(False).astype(bool).sum()
            else:
                row_features['emergency_stops_count'] = 0
                
            if 'safety_photocell_blocked' in window_logs:
                row_features['photocell_blocks_count'] = window_logs['safety_photocell_blocked'].fillna(False).astype(bool).sum()
            else:
                row_features['photocell_blocks_count'] = 0
        else:
            # If no logs fall in this window, fill with NaNs (Pipeline will impute later)
            row_features['motor_temp_mean'] = np.nan
            row_features['motor_temp_max'] = np.nan
            row_features['motor_torque_mean'] = np.nan
            row_features['motor_torque_max'] = np.nan
            row_features['emergency_stops_count'] = 0
            row_features['photocell_blocks_count'] = 0
            
        features_list.append(row_features)
        
    return pd.DataFrame(features_list)

# --- SECTION 4: DATASET INTEGRATION ---

def build_modeling_dataset(temporal_features_df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the dynamic temporal features with the static historical maintenance features.
    """
    print("Merging temporal and static features...")
    
    # We drop 'days_to_next_failure' from maintenance_df because we computed the accurate temporal one
    static_df = maintenance_df[['door_id', 'maintenance_type', 'number_of_past_failures']].copy()
    
    final_df = pd.merge(temporal_features_df, static_df, on='door_id', how='left')
    return final_df

# --- SECTION 5: MODEL TRAINING & INFERENCE ---

def train_evaluate_and_rank(df: pd.DataFrame, original_maintenance_path: str):
    """
    Splits data into historical (train/test) and current active states (inference).
    Trains the Random Forest, outputs the ranking, and saves the final predictions.
    """
    print("\n--- Training Predictive Model ---")
    
    # 1. Define Features
    numeric_features = [
        'motor_temp_mean', 'motor_temp_max', 'motor_torque_mean', 
        'motor_torque_max', 'emergency_stops_count', 'photocell_blocks_count',
        'number_of_past_failures'
    ]
    categorical_features = ['door_type', 'usage_scenario', 'installation_environment', 'maintenance_type']
    
    # 2. Split into Historical (has target) and Active (needs prediction)
    historical_data = df[df['days_to_next_failure'].notna()].copy()
    active_doors = df[df['days_to_next_failure'].isna()].copy()
    
    X_hist = historical_data[numeric_features + categorical_features]
    y_hist = historical_data['days_to_next_failure']
    X_active = active_doors[numeric_features + categorical_features]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.30, random_state=42)
    
    # 3. Build Pipeline
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
    
    # 4. Train & Evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("=== Regression Evaluation on Test Set ===")
    print(f"MAE  (days): {mean_absolute_error(y_test, y_pred):.3f}")
    # Fix for newer sklearn versions:
    print(f"RMSE (days): {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"R^2        : {r2_score(y_test, y_pred):.3f}")
    
    # 5. Inference & Ranking
    print("\n--- Generating Decision Support Output ---")
    active_doors['predicted_days_to_failure'] = model.predict(X_active)
    ranking = active_doors.sort_values("predicted_days_to_failure", ascending=True)
    
    cols_to_show = ['door_id', 'door_type', 'maintenance_type', 'predicted_days_to_failure']
    print(f"\n=== Top 10 Doors Predicted to Fail Sooner (Urgent Maintenance) ===")
    print(ranking[cols_to_show].head(10).to_string(index=False))
    
    # 6. SAVE PREDICTIONS BACK TO MAINTENANCE HISTORY
    print("\n--- Saving Final Predictions to Maintenance History ---")
    # Load the original maintenance history
    maintenance_update_df = pd.read_csv(original_maintenance_path)
    
    # Create a mapping dictionary: {door_id: predicted_days}
    prediction_map = dict(zip(active_doors['door_id'], active_doors['predicted_days_to_failure']))
    
    # Fill the 'days_to_next_failure' column
    maintenance_update_df['days_to_next_failure'] = maintenance_update_df['door_id'].map(prediction_map)
    
    # Export the final populated dataset
    output_path = Path("predicted_maintenance_history.csv")
    maintenance_update_df.to_csv(output_path, index=False)
    print(f"Predictions successfully merged and saved to: {output_path}")

# --- MAIN EXECUTION ---

def main():
    # 1. Load Data
    incidents_df, maintenance_df, all_logs_df = load_data("data")
    
    # 2. Build Targets (Now passing maintenance_df to catch missing doors)
    target_df = build_target_dataframe(incidents_df, maintenance_df)
    
    # 3. Extract Leakage-Free Features
    temporal_features_df = extract_windowed_features(target_df, all_logs_df)
    
    # 4. Build Final Dataset
    final_modeling_df = build_modeling_dataset(temporal_features_df, maintenance_df)
    
    # 5. Save the final formatted modeling table
    output_path = Path("regression_failure.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_modeling_df.to_csv(output_path, index=False)
    print(f"\nModeling dataset successfully saved to: {output_path}")
    
    # 6. Train and Rank (Now passing the path to original maintenance file)
    train_evaluate_and_rank(final_modeling_df, "../datasets/erreka_maintenance_history.csv")

if __name__ == "__main__":
    main()