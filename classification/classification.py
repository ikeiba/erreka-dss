import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Conection to mysql
def obtener_conexion():
    """Establece conexión con el Data Warehouse."""
    return create_engine("mysql+mysqlconnector://root:admin@localhost:3307/erreka_dss")

# Dataset construction
def construir_dataset_temporal(engine, fecha_corte, horas_pasado=48, es_entrenamiento=True):
    """Extrae logs pasados y mira incidentes futuros para el Target."""
    fecha_ref_dt = pd.to_datetime(fecha_corte)
    inicio_logs = (fecha_ref_dt - pd.Timedelta(hours=horas_pasado)).strftime('%Y-%m-%d %H:%M:%S')
    str_fecha_ref = fecha_ref_dt.strftime('%Y-%m-%d %H:%M:%S')

    # Unified logs
    query_logs = f"""
        SELECT door_id, door_type, usage_scenario, motor_torque, motor_temperature, emergency_stop_activated
        FROM garage_operations_log WHERE timestamp BETWEEN '{inicio_logs}' AND '{str_fecha_ref}'
        UNION ALL
        SELECT door_id, door_type, usage_scenario, motor_torque, motor_temperature, emergency_stop_activated
        FROM industrial_operations_log WHERE timestamp BETWEEN '{inicio_logs}' AND '{str_fecha_ref}'
        UNION ALL
        SELECT door_id, door_type, usage_scenario, motor_torque, motor_temperature, emergency_stop_activated
        FROM pedestrian_operations_log WHERE timestamp BETWEEN '{inicio_logs}' AND '{str_fecha_ref}'
    """
    logs_raw = pd.read_sql(query_logs, engine)
    
    logs_features = logs_raw.groupby('door_id').agg(
        door_type=('door_type', 'max'),
        usage_scenario=('usage_scenario', 'max'),
        total_ops=('motor_torque', 'count'),
        temp_max=('motor_temperature', 'max'),
        torque_mean=('motor_torque', 'mean'),
        paradas_emergencia=('emergency_stop_activated', 'sum')
    ).reset_index()

    # Previous failures (48 hours)
    query_incidents_past = f"""
        SELECT door_id, COUNT(*) as fallos_recientes
        FROM incident_events 
        WHERE timestamp BETWEEN '{inicio_logs}' AND '{str_fecha_ref}'
        GROUP BY door_id
    """
    incidents_past = pd.read_sql(query_incidents_past, engine)

    hist_df = pd.read_sql("SELECT * FROM erreka_maintenance_history", engine)

    # Data merging
    df_final = logs_features.merge(hist_df[['door_id', 'maintenance_type', 'number_of_past_failures']], on='door_id', how='left')
    df_final = df_final.merge(incidents_past, on='door_id', how='left')
    df_final.fillna(0, inplace=True)

    # Target (looking 30 days in the future)
    if es_entrenamiento:
        fin_target = (fecha_ref_dt + pd.Timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        target_df = pd.read_sql(f"SELECT DISTINCT door_id FROM incident_events WHERE timestamp BETWEEN '{str_fecha_ref}' AND '{fin_target}'", engine)
        df_final['target_falla'] = df_final['door_id'].apply(lambda x: 1 if x in target_df['door_id'].values else 0)
    
    return df_final, hist_df

# Training and evaluation
def entrenar_modelo(df_entrenamiento):
    """Entrena y evalúa con Split por puertas (GroupShuffleSplit)."""
    print("\n--- ENTRENANDO MODELO Y CALCULANDO ACCURACY ---")
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df_entrenamiento, groups=df_entrenamiento['door_id']))

    df_train, df_test = df_entrenamiento.iloc[train_idx], df_entrenamiento.iloc[test_idx]
    X_train, y_train = df_train.drop(columns=['door_id', 'target_falla']), df_train['target_falla']
    X_test, y_test = df_test.drop(columns=['door_id', 'target_falla']), df_test['target_falla']

    preprocesador = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['door_type', 'usage_scenario', 'maintenance_type']),
        ('num', 'passthrough', ['total_ops', 'temp_max', 'torque_mean', 'paradas_emergencia', 'number_of_past_failures', 'fallos_recientes'])
    ])

    pipeline = Pipeline([
        ('pre', preprocesador),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    print(f"Accuracy en Test: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds))

    return pipeline

# Final predictions and construction of final table
def categorizar_riesgo(prob):
    if prob >= 0.70: return 'High Risk'
    if prob >= 0.30: return 'Medium Risk'
    return 'Low Risk'

def generar_resultado_final(modelo, engine):
    """Toma la foto actual, predice riesgo y actualiza la tabla de mantenimiento."""
    print("\n--- GENERANDO PREDICCIONES PARA EL MANTENIMIENTO ---")
    
    # Present table
    df_actual, tabla_hist_completa = construir_dataset_temporal(
        engine, fecha_corte="2025-01-03 23:59:59", horas_pasado=48, es_entrenamiento=False
    )

    X_actual = df_actual.drop(columns=['door_id'])
    
    # Clasification and probability prediction
    probs = modelo.predict_proba(X_actual)[:, 1]
    preds_binarias = modelo.predict(X_actual)

    df_actual['failed_next_30_days'] = preds_binarias
    df_actual['probability_score'] = probs
    df_actual['risk_level'] = df_actual['probability_score'].apply(categorizar_riesgo)

    final_df = tabla_hist_completa.copy()
    
    # Clean original column
    if 'failed_next_30_days' in final_df.columns:
        final_df = final_df.drop(columns=['failed_next_30_days'])

    # Merge results
    final_df = final_df.merge(
        df_actual[['door_id', 'failed_next_30_days', 'probability_score', 'risk_level']], 
        on='door_id', how='left'
    )

    # Fill doors without logs (if exist)
    final_df['failed_next_30_days'] = final_df['failed_next_30_days'].fillna(0).astype(int)
    final_df['probability_score'] = final_df['probability_score'].fillna(0)
    final_df['risk_level'] = final_df['risk_level'].fillna('Low Risk')

    return final_df

# Execution
if __name__ == "__main__":
    motor = obtener_conexion()
    
    # Load and prepare data
    dataset_entreno, _ = construir_dataset_temporal(motor, "2025-01-01 23:59:59")
    
    # Train and evaluate model
    modelo_final = entrenar_modelo(dataset_entreno)
    
    # Generate unified table
    mantenimiento_actualizado = generar_resultado_final(modelo_final, motor)
    
    # Final results
    print("\n--- MUESTRA DE LA TABLA FINAL ACTUALIZADA ---")
    print(mantenimiento_actualizado[['door_id', 'failed_next_30_days', 'probability_score', 'risk_level']].sort_values(by='probability_score', ascending=False).head(10))

    # Store in a csv
    mantenimiento_actualizado.to_csv("../datasets/erreka_maintenance_history_updated.csv", index=False)