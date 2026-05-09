import os

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MYSQL_HOST     = "localhost"
MYSQL_PORT     = 3306
MYSQL_USER     = "root"
MYSQL_PASSWORD = os.getenv("ROOT_PASSWORD")
MYSQL_DB       = os.getenv("MYSQL_DATABASE", "erreka_dss")

CONTAMINATION  = 0.10  # fracción esperada de puertas anómalas (~10%)

engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)
print("Connection to MySQL established successfully.")


# ─────────────────────────────────────────────
# 1. EXTRACT — leer logs desde MySQL
# ─────────────────────────────────────────────
print("\nReading operational logs from MySQL...")

ped = pd.read_sql("SELECT * FROM pedestrian_operations_log", engine, parse_dates=["timestamp"])
ind = pd.read_sql("SELECT * FROM industrial_operations_log",  engine, parse_dates=["timestamp"])
gar = pd.read_sql("SELECT * FROM garage_operations_log",      engine, parse_dates=["timestamp"])


# ─────────────────────────────────────────────
# 2. TRANSFORM — agregar métricas por puerta
#    → door_anomaly_metrics (estructura requerida por la actividad)
# ─────────────────────────────────────────────
print("\nAggregating metrics per door...")

# Columnas de sensores de cada log
PED_SENSORS = ["motion_radar_detected", "pir_presence_detected", "presence_ir_active",
               "pressure_mat_activated", "safety_photocell_blocked", "infrared_curtain_triggered"]
IND_SENSORS = ["anti_fall_triggered", "industrial_radar_detected", "inductive_loop_triggered",
               "photoelectric_barrier_blocked", "infrared_curtain_triggered", "bottom_sensitive_edge_triggered"]
GAR_SENSORS = [c for c in gar.columns if any(k in c for k in ["radar","sensor","photocell","curtain","loop","mat"])]

def add_sensor_flag(df, sensor_cols):
    existing = [c for c in sensor_cols if c in df.columns]
    df["sensor_activated"] = df[existing].any(axis=1).astype(int)
    return df

ped = add_sensor_flag(ped, PED_SENSORS)
ind = add_sensor_flag(ind, IND_SENSORS)
gar = add_sensor_flag(gar, GAR_SENSORS)

# Unificar sólo las columnas necesarias
KEEP = ["door_id", "timestamp", "cycle_counter", "motor_temperature",
        "vibration_level", "motor_torque", "sensor_activated"]

def safe_select(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()

ops = pd.concat([safe_select(ped, KEEP),
                 safe_select(ind, KEEP),
                 safe_select(gar, KEEP)], ignore_index=True)

# Ciclos por día → avg y peak
ops["date"] = ops["timestamp"].dt.date
cycles_per_day = (ops.groupby(["door_id", "date"])["cycle_counter"]
                     .max().reset_index()
                     .rename(columns={"cycle_counter": "cycles_that_day"}))

agg_cycles = cycles_per_day.groupby("door_id")["cycles_that_day"].agg(
    avg_cycles_per_day="mean",
    peak_cycles="max"
).reset_index()

agg_ops = ops.groupby("door_id").agg(
    avg_motor_temp         = ("motor_temperature", "mean"),
    temp_std               = ("motor_temperature", "std"),
    avg_vibration          = ("vibration_level",   "mean"),
    sensor_activation_rate = ("sensor_activated",  "mean"),
).reset_index()

metrics = agg_cycles.merge(agg_ops, on="door_id", how="inner")
metrics["temp_std"] = metrics["temp_std"].fillna(0)

print(f"Doors with metrics: {len(metrics)}")


# ─────────────────────────────────────────────
# 3. MODEL — Isolation Forest
# ─────────────────────────────────────────────
print("\nTraining Isolation Forest...")

FEATURES = ["avg_cycles_per_day", "peak_cycles", "avg_motor_temp",
            "temp_std", "avg_vibration", "sensor_activation_rate"]

X_scaled = StandardScaler().fit_transform(metrics[FEATURES])

model = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42)
model.fit(X_scaled)

raw_scores  = model.decision_function(X_scaled)
predictions = model.predict(X_scaled)

# Normalizar score a [0, 100] — 100 = más anómala
min_s, max_s = raw_scores.min(), raw_scores.max()
metrics["anomaly_score"] = (100 * (1 - (raw_scores - min_s) / (max_s - min_s))).round(1)
metrics["anomaly_flag"]  = (predictions == -1).astype(int)   # 1 = anómala
metrics["anomaly_rank"]  = metrics["anomaly_score"].rank(ascending=False).astype(int)

def risk_label(row):
    if row["anomaly_flag"] == 1:
        return "Critical" if row["anomaly_score"] >= 85 else "High"
    return "Medium" if row["anomaly_score"] >= 60 else "Normal"

metrics["risk_level"] = metrics.apply(risk_label, axis=1)
metrics = metrics.sort_values("anomaly_rank").reset_index(drop=True)

print(f"\n  Anomaly distribution:")
print(metrics["risk_level"].value_counts().to_string())
print(f"\n  Top 10 most anomalous doors:")
print(metrics[["door_id", "anomaly_score", "risk_level"]].head(10).to_string(index=False))


# ─────────────────────────────────────────────
# 4. LOAD — escribir resultados en MySQL
# ─────────────────────────────────────────────
print("\nWriting results to MySQL...")

# Tabla de métricas agregadas (requerida por la actividad)
door_anomaly_metrics = metrics[["door_id", "avg_cycles_per_day", "peak_cycles",
                                 "avg_motor_temp", "temp_std", "avg_vibration",
                                 "sensor_activation_rate"]]

# Tabla de resultados del modelo (para Grafana)
door_anomaly_results = metrics[["door_id", "anomaly_score", "anomaly_flag",
                                 "anomaly_rank", "risk_level",
                                 "avg_cycles_per_day", "peak_cycles",
                                 "avg_motor_temp", "temp_std",
                                 "avg_vibration", "sensor_activation_rate"]]

for table_name, df_out in [("door_anomaly_metrics", door_anomaly_metrics),
                            ("door_anomaly_results", door_anomaly_results)]:
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`;"))
    df_out.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"{table_name} — {len(df_out)} rows written")

print("\nDone. Tables ready for Grafana:")
print("   → door_anomaly_metrics (aggregated features)")
print("   → door_anomaly_results (anomaly scores + ranking)")
