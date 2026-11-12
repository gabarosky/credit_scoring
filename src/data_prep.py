import duckdb
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(ROOT_DIR, "data", "credit_scoring.db")
CSV_PATH = os.path.join(ROOT_DIR, "data", "raw", "cs-training.csv")

def ingest_csv_to_duckdb(csv_path: str, db_path: str, table_name: str) -> None:
    """Lee un CSV y lo persiste como tabla en una base de datos DuckDB local."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_csv_auto('{csv_path}', header=true);
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"Tabla '{table_name}' creada en '{db_path}' con {count:,} registros.")
    con.close()

def build_stg(db_path: str = DB_PATH) -> None:
    """raw → stg: rename a snake_case, cast 'NA'→NULL. Sin filtros de filas."""
    con = duckdb.connect(db_path)
    try:
        con.execute("""
            CREATE OR REPLACE TABLE stg_credit_applications AS
            SELECT
                "SeriousDlqin2yrs"                                  AS target,
                "RevolvingUtilizationOfUnsecuredLines"              AS revolving_utilization,
                age,
                "NumberOfTime30-59DaysPastDueNotWorse"              AS times_30_59_dpd,
                "DebtRatio"                                         AS debt_ratio,
                CAST(NULLIF("MonthlyIncome", 'NA') AS DOUBLE)       AS monthly_income,
                "NumberOfOpenCreditLinesAndLoans"                   AS open_credit_lines,
                "NumberOfTimes90DaysLate"                           AS times_90_dpd,
                "NumberRealEstateLoansOrLines"                      AS real_estate_loans,
                "NumberOfTime60-89DaysPastDueNotWorse"              AS times_60_89_dpd,
                CAST(NULLIF("NumberOfDependents", 'NA') AS INTEGER) AS dependents
            FROM raw_credit_applications
        """)
    finally:
        con.close()