import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ---------------------------
# CONFIGURATION
# ---------------------------
DB_USER     = "root"
DB_PASSWORD = os.getenv("ROOT_PASSWORD")
DB_HOST     = "localhost"
DB_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
DB_NAME     = os.getenv("MYSQL_DATABASE")

OUTPUT_CSV   = Path("./erreka_door_behavior_metrics.csv")
TARGET_TABLE = "erreka_door_behavior_metrics"

DB_CONFIG = {
    "host":     DB_HOST,
    "port":     DB_PORT,
    "user":     DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
}

LOG_SOURCES = {
    "pedestrian_operations_log": [
        "motion_radar_detected",
        "pir_presence_detected",
        "presence_ir_active",
        "pressure_mat_activated",
        "safety_photocell_blocked",
        "infrared_curtain_triggered",
    ],
    "industrial_operations_log": [
        "industrial_radar_detected",
        "inductive_loop_triggered",
        "photoelectric_barrier_blocked",
        "infrared_curtain_triggered",
        "bottom_sensitive_edge_triggered",
        "anti_fall_triggered",
    ],
    "garage_operations_log": [
        "safety_photocell_blocked",
        "sensitive_edge_triggered",
    ],
}

METRIC_COLUMNS = [
    "door_id",
    "avg_cycles_per_day",
    "peak_usage_cycles",
    "avg_motor_temp",
    "temp_variability",
    "sensor_activation_rate",
    "environment",
    "door_type",
]

NUMERIC_METRIC_COLUMNS = [
    "avg_cycles_per_day",
    "peak_usage_cycles",
    "avg_motor_temp",
    "temp_variability",
    "sensor_activation_rate",
]


def build_metrics_query(log_table, sensor_columns):
    sensor_or = " OR ".join(f"`{col}` = 1" for col in sensor_columns)
    return f"""
        WITH daily_cycles AS (
            SELECT
                door_id,
                DATE(timestamp) AS day,
                SUM(CASE WHEN motor_command = 'open' THEN 1 ELSE 0 END) AS cycles_that_day
            FROM `{log_table}`
            GROUP BY door_id, DATE(timestamp)
        ),
        cycle_agg AS (
            SELECT
                door_id,
                SUM(cycles_that_day) AS total_cycles,
                COUNT(*)             AS active_days,
                MAX(cycles_that_day) AS peak_usage_cycles
            FROM daily_cycles
            GROUP BY door_id
        ),
        overall_agg AS (
            SELECT
                door_id,
                AVG(motor_temperature)         AS avg_motor_temp,
                STDDEV_SAMP(motor_temperature) AS temp_variability,
                SUM(CASE WHEN {sensor_or} THEN 1 ELSE 0 END) AS sensor_activated_rows,
                COUNT(*)                       AS total_rows
            FROM `{log_table}`
            GROUP BY door_id
        )
        SELECT
            c.door_id,
            ROUND((c.total_cycles / c.active_days), 1)         AS avg_cycles_per_day,
            c.peak_usage_cycles                                 AS peak_usage_cycles,
            ROUND(o.avg_motor_temp, 1)                          AS avg_motor_temp,
            ROUND(o.temp_variability, 2)                        AS temp_variability,
            ROUND((o.sensor_activated_rows / o.total_rows), 3)  AS sensor_activation_rate
        FROM cycle_agg c
        JOIN overall_agg o ON c.door_id = o.door_id
    """


def fetch_log_metrics(cursor, log_table, sensor_columns):
    print(f"\nAggregating {log_table}...")
    cursor.execute(build_metrics_query(log_table, sensor_columns))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

    for col in ("avg_cycles_per_day", "avg_motor_temp", "temp_variability", "sensor_activation_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["peak_usage_cycles"] = pd.to_numeric(df["peak_usage_cycles"], errors="coerce").astype(float)

    print(f"  ✓ {len(df)} doors aggregated from {log_table}")
    return df


def fetch_registry_context(cursor):
    print("\nFetching doors_registry context...")
    cursor.execute("""
        SELECT
            door_id,
            installation_environment AS environment,
            door_type
        FROM doors_registry
    """)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    print(f"  doors_registry rows: {len(df)}")
    return df


def create_target_table(cursor):
    print(f"\nRecreating target table `{TARGET_TABLE}`...")
    cursor.execute(f"DROP TABLE IF EXISTS `{TARGET_TABLE}`")
    cursor.execute(f"""
        CREATE TABLE `{TARGET_TABLE}` (
            `door_id`                VARCHAR(20)   NOT NULL,
            `avg_cycles_per_day`     DECIMAL(8,1)  NULL,
            `peak_usage_cycles`      DECIMAL(8,1)  NULL,
            `avg_motor_temp`         DECIMAL(5,1)  NULL,
            `temp_variability`       DECIMAL(5,2)  NULL,
            `sensor_activation_rate` DECIMAL(5,3)  NULL,
            `environment`            VARCHAR(50)   NULL,
            `door_type`              VARCHAR(50)   NULL,
            PRIMARY KEY (`door_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    print(f"  ✓ Target table ready.")


def insert_metrics(connection, cursor, df):
    print(f"\nInserting {len(df)} rows into `{TARGET_TABLE}`...")
    columns_str  = ", ".join(f"`{c}`" for c in METRIC_COLUMNS)
    placeholders = ", ".join(["%s"] * len(METRIC_COLUMNS))
    insert_query = f"INSERT INTO `{TARGET_TABLE}` ({columns_str}) VALUES ({placeholders})"

    data = []
    for row in df[METRIC_COLUMNS].to_numpy():
        cleaned_row = tuple(None if pd.isna(v) else v for v in row)
        data.append(cleaned_row)

    cursor.executemany(insert_query, data)
    connection.commit()
    print(f"  ✓ Inserted {len(data)} rows.")


def validate(cursor):
    print("\n" + "=" * 60)
    print("Post-load Validation")
    print("=" * 60)

    cursor.execute(f"SELECT COUNT(*) FROM `{TARGET_TABLE}`")
    total = cursor.fetchone()[0]
    print(f"Total rows in {TARGET_TABLE}: {total}")

    print("\nRow count per door_type:")
    cursor.execute(f"""
        SELECT COALESCE(door_type, '(NULL)') AS door_type, COUNT(*)
        FROM `{TARGET_TABLE}` GROUP BY door_type ORDER BY door_type
    """)
    for dt, n in cursor.fetchall():
        print(f"  {dt}: {n}")

    print("\nRow count per environment:")
    cursor.execute(f"""
        SELECT COALESCE(environment, '(NULL)') AS environment, COUNT(*)
        FROM `{TARGET_TABLE}` GROUP BY environment ORDER BY environment
    """)
    for env, n in cursor.fetchall():
        print(f"  {env}: {n}")

    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT door_id FROM pedestrian_operations_log
            UNION
            SELECT door_id FROM industrial_operations_log
            UNION
            SELECT door_id FROM garage_operations_log
        ) t
    """)
    expected = cursor.fetchone()[0]
    print(f"\nExpected distinct door_id across logs: {expected}")
    print(f"Actual rows in {TARGET_TABLE}:           {total}")
    if expected == total:
        print("  ✓ Row count matches.")
    else:
        print(f"  ✗ MISMATCH: delta = {total - expected}")


def describe_numeric(df):
    print("\n" + "=" * 60)
    print("Descriptive stats")
    print("=" * 60)
    for col in NUMERIC_METRIC_COLUMNS:
        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(series.isna().sum())
        non_nan = series.dropna()
        if len(non_nan) == 0:
            print(f"  {col:>25s}: all NaN  (NaN={nan_count})")
            continue
        print(
            f"  {col:>25s}: "
            f"min={float(non_nan.min()):12.4f}  "
            f"max={float(non_nan.max()):12.4f}  "
            f"mean={float(non_nan.mean()):12.4f}  "
            f"NaN={nan_count}"
        )


def main():
    print("\n" + "=" * 60)
    print("ERREKA DSS - Behavioral Metrics Build")
    print("=" * 60)

    connection = None
    cursor = None

    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SET SESSION wait_timeout=600")
        cursor.execute("SET SESSION interactive_timeout=600")
        print("Connection to MySQL established successfully.")

        per_log_frames = []
        for log_table, sensor_columns in LOG_SOURCES.items():
            per_log_frames.append(fetch_log_metrics(cursor, log_table, sensor_columns))

        metrics = pd.concat(per_log_frames, ignore_index=True)
        print(f"\nTotal aggregated rows (before join): {len(metrics)}")

        duplicate_doors = metrics["door_id"].duplicated().sum()
        if duplicate_doors > 0:
            print(f"  WARNING: {duplicate_doors} door_id(s) appear in more than one log.")

        registry = fetch_registry_context(cursor)
        merged   = metrics.merge(registry, on="door_id", how="left")

        missing_count = int((~merged["door_id"].isin(set(registry["door_id"]))).sum())
        if missing_count > 0:
            print(f"  WARNING: {missing_count} door(s) missing from doors_registry; kept as NULL.")

        create_target_table(cursor)
        insert_metrics(connection, cursor, merged)
        validate(cursor)

        print(f"\nExporting CSV to {OUTPUT_CSV}...")
        merged[METRIC_COLUMNS].to_csv(OUTPUT_CSV, index=False)
        print(f"  ✓ Wrote {len(merged)} rows to {OUTPUT_CSV}")

        describe_numeric(merged)

        print("\n" + "=" * 60)
        print("Build Complete")
        print("=" * 60)

    except Error as e:
        print(f"\n✗ MySQL Error: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection closed.")


if __name__ == "__main__":
    main()