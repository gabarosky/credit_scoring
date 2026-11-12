from pathlib import Path
import duckdb

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "credit_scoring.db"

# --- Caps (constantes versionadas, ver decisions.md) ---
CAPS = {
    "revolving_utilization": 2.0,
    "times_30_59_dpd":       4,
    "times_60_89_dpd":       4,
    "times_90_dpd":          4,
    "debt_ratio":            3.0,
    "open_credit_lines":     20,
    "real_estate_loans":     5,
    "dependents":            4,
}

TARGET     = "target"
STG_TABLE  = "stg_credit_applications"
FEAT_TABLE = "features_credit_applications"


def _get_con(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def _build_features_sql(caps: dict) -> str:
    return f"""
        CREATE OR REPLACE TABLE {FEAT_TABLE} AS
        WITH base AS (
            SELECT
                target,
                revolving_utilization,
                age,
                times_30_59_dpd,
                times_60_89_dpd,
                times_90_dpd,
                debt_ratio,
                monthly_income,
                open_credit_lines,
                real_estate_loans,
                dependents,
                CASE WHEN monthly_income IS NULL THEN 1 ELSE 0 END AS income_missing_flag,
                CASE WHEN dependents     IS NULL THEN 1 ELSE 0 END AS dependents_missing_flag,
                CASE WHEN debt_ratio > {caps['debt_ratio']}
                      AND (monthly_income IS NULL OR monthly_income = 0)
                     THEN 1 ELSE 0 END AS high_debt_missing_income_flag
            FROM {STG_TABLE}
            WHERE age > 0
        )
        SELECT
            target,
            LEAST(revolving_utilization, {caps['revolving_utilization']}) AS revolving_utilization,
            age,
            LEAST(times_30_59_dpd, {caps['times_30_59_dpd']}) AS times_30_59_dpd,
            LEAST(times_60_89_dpd, {caps['times_60_89_dpd']}) AS times_60_89_dpd,
            LEAST(times_90_dpd,    {caps['times_90_dpd']})    AS times_90_dpd,
            LEAST(debt_ratio,      {caps['debt_ratio']})      AS debt_ratio,
            monthly_income,
            LEAST(open_credit_lines, {caps['open_credit_lines']}) AS open_credit_lines,
            LEAST(real_estate_loans, {caps['real_estate_loans']}) AS real_estate_loans,
            CASE WHEN dependents IS NULL THEN NULL
                 ELSE LEAST(dependents, {caps['dependents']}) END AS dependents,
            income_missing_flag,
            dependents_missing_flag,
            high_debt_missing_income_flag,
            COALESCE(times_30_59_dpd, 0)
                + COALESCE(times_60_89_dpd, 0)
                + COALESCE(times_90_dpd, 0) AS total_past_due
        FROM base
    """


def build_features(caps: dict | None = None) -> None:
    """stg → features. Aplica caps, crea flags y `total_past_due`.
    NO imputa (va en pipeline). NO winsoriza monthly_income (va en pipeline)."""
    con = _get_con()
    try:
        con.execute(_build_features_sql(caps or CAPS))
    finally:
        con.close()


def get_feature_columns() -> list[str]:
    con = _get_con(read_only=True)
    try:
        cols = con.execute(f"SELECT * FROM {FEAT_TABLE} LIMIT 0").df().columns.tolist()
    finally:
        con.close()
    return [c for c in cols if c != TARGET]