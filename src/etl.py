"""Extract -> validate -> load pipeline for the AP / P-card ledger.

The pipeline reads the raw fiscal-year extracts and master files, applies
data-quality rules (rows that fail are quarantined with a reason rather than
silently dropped), loads the clean data into SQLite and creates the SQL views
used by the anomaly checks (see src/sql/views.sql).

Usage:
    python src/etl.py                 # builds data/warehouse/audit.db
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "warehouse" / "audit.db"
VIEWS_SQL = Path(__file__).resolve().parent / "sql" / "views.sql"

TXN_COLUMNS = ["txn_id", "voucher_no", "source", "invoice_no", "invoice_date", "posting_date",
               "vendor_id", "dept_id", "account_code", "requester_id", "approver_id", "amount"]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_txn_req_vendor ON transactions(requester_id, vendor_id, invoice_date)",
    "CREATE INDEX IF NOT EXISTS ix_txn_posting ON transactions(posting_date)",
    "CREATE INDEX IF NOT EXISTS ix_txn_amount ON transactions(amount)",
    "CREATE INDEX IF NOT EXISTS ix_txn_voucher ON transactions(voucher_no)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_txn_id ON transactions(txn_id)",
]


@dataclass
class ETLResult:
    rows_extracted: int
    rows_loaded: int
    rows_rejected: int
    dq_report: pd.DataFrame
    timings: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #

def extract(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read every fiscal-year extract (as text, so validation sees the raw values) plus masters."""
    files = sorted(raw_dir.glob("transactions_FY*.csv"))
    if not files:
        raise FileNotFoundError(f"No transaction extracts in {raw_dir}; run src/generate_data.py first")
    txns = pd.concat(
        [pd.read_csv(f, dtype=str, keep_default_na=False).assign(source_file=f.name) for f in files],
        ignore_index=True,
    )
    masters = {name: pd.read_csv(raw_dir / f"{name}.csv", dtype=str)
               for name in ("vendors", "departments", "employees", "accounts", "holidays")}
    return {"transactions": txns, **masters}


# --------------------------------------------------------------------------- #
# Validate / transform
# --------------------------------------------------------------------------- #

def normalise_invoice(s: pd.Series) -> pd.Series:
    """Upper-case and strip punctuation/whitespace so 'inv-001234 ' and 'INV001234' match."""
    return s.str.upper().map(lambda x: re.sub(r"[^A-Z0-9]", "", x))


def validate(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply data-quality rules. Returns (clean, rejects, dq_report)."""
    t = data["transactions"].copy()
    report = []

    def log(rule, n, action):
        report.append({"rule": rule, "rows_affected": int(n), "action": action})

    log("rows extracted", len(t), "-")

    # 1. standardise keys (fixable defects)
    for col in ("vendor_id", "dept_id", "account_code", "requester_id", "approver_id"):
        cleaned = t[col].str.strip().str.upper()
        changed = (cleaned != t[col]).sum()
        if changed:
            log(f"{col}: whitespace / case standardised", changed, "fixed")
        t[col] = cleaned

    # 2. exact duplicate extract records
    dup = t.duplicated(subset=TXN_COLUMNS, keep="first")
    log("exact duplicate records", dup.sum(), "removed")
    t = t[~dup]

    # 3. typed fields
    t["amount_num"] = pd.to_numeric(t["amount"].replace("", None), errors="coerce")
    t["posting_dt"] = pd.to_datetime(t["posting_date"], format="%Y-%m-%d", errors="coerce")
    t["invoice_dt"] = pd.to_datetime(t["invoice_date"], format="%Y-%m-%d", errors="coerce")

    vendors = set(data["vendors"]["vendor_id"])
    depts = set(data["departments"]["dept_id"])
    accounts = set(data["accounts"]["account_code"])
    emps = set(data["employees"]["employee_id"])

    rules = {
        "missing or non-numeric amount": t["amount_num"].isna(),
        "unparseable posting date": t["posting_dt"].isna(),
        "unparseable invoice date": t["invoice_dt"].isna(),
        "missing approver": t["approver_id"].eq(""),
        "unknown vendor id": ~t["vendor_id"].isin(vendors),
        "unknown department": ~t["dept_id"].isin(depts),
        "unknown GL account": ~t["account_code"].isin(accounts),
        "unknown requester": ~t["requester_id"].isin(emps),
        "unknown approver (non-blank)": t["approver_id"].ne("") & ~t["approver_id"].isin(emps),
        "duplicate txn_id with different content": t["txn_id"].duplicated(keep=False),
    }
    reason = pd.Series("", index=t.index)
    for name, mask in rules.items():
        log(name, mask.sum(), "quarantined")
        reason = reason.where(~mask | reason.ne(""), name)

    bad = reason.ne("")
    rejects = t.loc[bad, TXN_COLUMNS + ["source_file"]].assign(reject_reason=reason[bad])
    clean = t.loc[~bad].copy()

    clean["amount"] = clean["amount_num"].round(2)
    clean["posting_date"] = clean["posting_dt"].dt.strftime("%Y-%m-%d")
    clean["invoice_date"] = clean["invoice_dt"].dt.strftime("%Y-%m-%d")
    clean["fiscal_year"] = clean["posting_dt"].dt.year + (clean["posting_dt"].dt.month >= 7).astype(int)
    clean["invoice_key"] = normalise_invoice(clean["invoice_no"])
    clean["txn_id"] = clean["txn_id"].astype("int64")
    clean = clean[TXN_COLUMNS + ["fiscal_year", "invoice_key", "source_file"]]
    log("rows passed to load", len(clean), "-")
    return clean.reset_index(drop=True), rejects.reset_index(drop=True), pd.DataFrame(report)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load(conn: sqlite3.Connection, clean: pd.DataFrame, rejects: pd.DataFrame,
         data: dict[str, pd.DataFrame], create_indexes: bool = True) -> None:
    clean.to_sql("transactions", conn, if_exists="replace", index=False, chunksize=50_000)
    rejects.to_sql("rejected_transactions", conn, if_exists="replace", index=False)
    for name in ("vendors", "departments", "employees", "accounts", "holidays"):
        data[name].to_sql(name, conn, if_exists="replace", index=False)
    if create_indexes:
        for stmt in INDEXES:
            conn.execute(stmt)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_vendor ON vendors(vendor_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_hol ON holidays(holiday_date)")
    conn.executescript(VIEWS_SQL.read_text())
    conn.commit()


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    return conn


def run_etl(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH, create_indexes: bool = True) -> ETLResult:
    timings = {}
    t0 = time.perf_counter()
    data = extract(raw_dir)
    timings["extract_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    clean, rejects, dq = validate(data)
    timings["validate_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as conn:
        load(conn, clean, rejects, data, create_indexes=create_indexes)
    conn.close()
    timings["load_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0
    return ETLResult(len(data["transactions"]), len(clean), len(rejects), dq, timings)


def append_extract(extract_file: Path, db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR) -> ETLResult:
    """Incremental load: validate a single new extract (e.g. one month) and append it to an
    existing warehouse instead of rebuilding from every historical file. Rows whose txn_id is
    already loaded are skipped, so re-delivered extracts are idempotent."""
    timings = {}
    t0 = time.perf_counter()
    data = {"transactions": pd.read_csv(extract_file, dtype=str, keep_default_na=False)
            .assign(source_file=Path(extract_file).name)}
    data.update({name: pd.read_csv(raw_dir / f"{name}.csv", dtype=str)
                 for name in ("vendors", "departments", "employees", "accounts", "holidays")})
    timings["extract_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    clean, rejects, dq = validate(data)
    timings["validate_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    conn = connect(db_path)
    existing = set(pd.read_sql_query("SELECT txn_id FROM transactions", conn)["txn_id"])
    clean = clean[~clean["txn_id"].isin(existing)]
    clean.to_sql("transactions", conn, if_exists="append", index=False, chunksize=50_000)
    rejects.to_sql("rejected_transactions", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    timings["load_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0
    return ETLResult(len(data["transactions"]), len(clean), len(rejects), dq, timings)


if __name__ == "__main__":
    res = run_etl()
    print(res.dq_report.to_string(index=False))
    print({k: round(v, 2) for k, v in res.timings.items()})
