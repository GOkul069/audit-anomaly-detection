"""The nine automated anomaly checks.

Every check takes an open SQLite connection to the warehouse built by src/etl.py and
returns a DataFrame with one row per flagged transaction: a ``txn_id`` column plus
check-specific evidence columns. Seven checks are thin wrappers over SQL views
(src/sql/views.sql); the Benford and vendor z-score checks are statistical tests run
in pandas/scipy on top of the ``v_txn`` view.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

BENFORD_P = np.log10(1 + 1 / np.arange(1, 10))


def _view(conn: sqlite3.Connection, view: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {view}", conn)


# 1 ------------------------------------------------------------------------- #
def check_duplicate_payments(conn) -> pd.DataFrame:
    """Same vendor, amount and normalised invoice number paid more than once."""
    return _view(conn, "v_chk_duplicate_payments")


# 2 ------------------------------------------------------------------------- #
def check_split_purchases(conn) -> pd.DataFrame:
    """Clusters of just-under-threshold payments that together exceed an approval threshold."""
    df = _view(conn, "v_chk_split_purchases")
    # a payment can sit near both thresholds only in theory; keep one row per txn
    return df.sort_values("threshold", ascending=False).drop_duplicates("txn_id").reset_index(drop=True)


# 3 ------------------------------------------------------------------------- #
def check_weekend_holiday(conn) -> pd.DataFrame:
    """AP vouchers posted on a Saturday, Sunday or university holiday."""
    return _view(conn, "v_chk_weekend_holiday")


# 4 ------------------------------------------------------------------------- #
def benford_table(amounts: pd.Series) -> tuple[np.ndarray, int]:
    a = amounts.abs()
    a = a[a >= 10]
    first = (a / 10 ** np.floor(np.log10(a))).astype(int).clip(1, 9)
    counts = np.bincount(first, minlength=10)[1:10]
    return counts, int(counts.sum())


def benford_by_group(df: pd.DataFrame, group_col: str, min_n: int = 100) -> pd.DataFrame:
    """First-digit Benford test per group: mean absolute deviation (Nigrini) and chi-square."""
    rows = []
    for key, g in df.groupby(group_col):
        counts, n = benford_table(g["amount"])
        if n < min_n:
            continue
        obs = counts / n
        mad = float(np.abs(obs - BENFORD_P).mean())
        chi2, p = stats.chisquare(counts, BENFORD_P * n)
        rows.append({group_col: key, "n": n, "mad": mad, "chi2": float(chi2), "p_value": float(p)})
    return pd.DataFrame(rows)


def check_benford(conn, group_col: str = "requester_id", mad_threshold: float = 0.015,
                  alpha: float = 0.001, min_n: int = 100) -> pd.DataFrame:
    """Requesters whose amounts do not follow Benford's law (MAD above Nigrini's 0.015
    non-conformity cut-off and chi-square p < alpha). Returns all of their transactions."""
    df = pd.read_sql_query(f"SELECT txn_id, {group_col}, amount FROM transactions", conn)
    tests = benford_by_group(df, group_col, min_n)
    bad = tests[(tests["mad"] > mad_threshold) & (tests["p_value"] < alpha)]
    out = df.merge(bad[[group_col, "n", "mad", "p_value"]], on=group_col)
    return out[["txn_id", group_col, "mad", "p_value"]].reset_index(drop=True)


# 5 ------------------------------------------------------------------------- #
def check_round_dollar(conn, exclude_groups: tuple[str, ...] = ("Student Support",)) -> pd.DataFrame:
    """Exact multiples of $1,000 (>= $1,000), excluding account groups that are paid in fixed
    award amounts by design (stipends and fellowships)."""
    df = _view(conn, "v_chk_round_dollar")
    return df[~df["account_group"].isin(exclude_groups)].reset_index(drop=True)


# 6 ------------------------------------------------------------------------- #
def check_vendor_outliers(conn, z_threshold: float = 4.0, min_history: int = 20) -> pd.DataFrame:
    """Payments far above a vendor's normal range: robust z-score of log(amount) using the
    vendor median and MAD, so the outliers themselves do not inflate the baseline."""
    df = pd.read_sql_query(
        "SELECT txn_id, vendor_id, amount FROM transactions WHERE amount > 0 AND source = 'AP'", conn)
    df["log_amt"] = np.log(df["amount"])
    g = df.groupby("vendor_id")["log_amt"]
    df["n_vendor"] = g.transform("size")
    df["median"] = g.transform("median")
    df["mad"] = (df["log_amt"] - df["median"]).abs().groupby(df["vendor_id"]).transform("median")
    df["robust_z"] = 0.6745 * (df["log_amt"] - df["median"]) / df["mad"].replace(0, np.nan)
    flagged = df[(df["n_vendor"] >= min_history) & (df["robust_z"] > z_threshold)]
    flagged = flagged.assign(vendor_median_amount=np.exp(flagged["median"]).round(2))
    return flagged[["txn_id", "vendor_id", "amount", "vendor_median_amount", "robust_z"]].reset_index(drop=True)


# 7 ------------------------------------------------------------------------- #
def check_new_vendor_large(conn) -> pd.DataFrame:
    """Payments of $25,000+ to a vendor set up less than 90 days earlier."""
    return _view(conn, "v_chk_new_vendor_large")


# 8 ------------------------------------------------------------------------- #
def check_self_approval(conn) -> pd.DataFrame:
    """Requester approved their own transaction."""
    return _view(conn, "v_chk_self_approval")


# 9 ------------------------------------------------------------------------- #
def check_sequence_gaps(conn) -> pd.DataFrame:
    """Missing voucher numbers in the AP sequence; the voucher after each gap is flagged."""
    return _view(conn, "v_chk_sequence_gaps")


CHECKS: dict[str, Callable[[sqlite3.Connection], pd.DataFrame]] = {
    "duplicate_payment": check_duplicate_payments,
    "split_purchase": check_split_purchases,
    "weekend_holiday_posting": check_weekend_holiday,
    "benford_deviation": check_benford,
    "round_dollar": check_round_dollar,
    "vendor_amount_outlier": check_vendor_outliers,
    "new_vendor_large_payment": check_new_vendor_large,
    "self_approval": check_self_approval,
    "sequence_gap": check_sequence_gaps,
}

CHECK_DESCRIPTIONS = {
    "duplicate_payment": "Same vendor, amount and normalised invoice number paid more than once",
    "split_purchase": "Near-threshold payments by one requester to one vendor within 7 days that together exceed $5k/$25k",
    "weekend_holiday_posting": "AP voucher posted on a weekend or university holiday",
    "benford_deviation": "Requester's amounts fail the Benford first-digit test (MAD > 0.015, p < 0.001)",
    "round_dollar": "Exact multiple of $1,000 (stipend / fellowship accounts excluded)",
    "vendor_amount_outlier": "Robust z-score of log(amount) > 4 within the vendor's history",
    "new_vendor_large_payment": "$25k+ paid to a vendor created < 90 days before posting",
    "self_approval": "Requester approved their own transaction",
    "sequence_gap": "Voucher immediately after a missing AP voucher number",
}

# Weight used to rank exceptions for analyst triage (higher = review first)
CHECK_WEIGHTS = {
    "duplicate_payment": 5, "self_approval": 5, "new_vendor_large_payment": 5, "split_purchase": 4,
    "vendor_amount_outlier": 3, "benford_deviation": 3, "sequence_gap": 2,
    "weekend_holiday_posting": 2, "round_dollar": 1,
}


def run_all_checks(conn) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the nine checks; returns (results by check, timing table)."""
    results, timing = {}, []
    for name, fn in CHECKS.items():
        t0 = time.perf_counter()
        res = fn(conn)
        timing.append({"check": name, "flagged_rows": len(res), "seconds": time.perf_counter() - t0})
        results[name] = res
    return results, pd.DataFrame(timing)


def build_exceptions_report(conn, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per flagged transaction with the checks it failed and a triage priority score."""
    long = pd.concat([r[["txn_id"]].assign(check=name) for name, r in results.items()], ignore_index=True)
    long = long.drop_duplicates()
    agg = long.groupby("txn_id").agg(
        checks_failed=("check", lambda s: "; ".join(sorted(s))),
        n_checks=("check", "size"),
        risk_score=("check", lambda s: sum(CHECK_WEIGHTS[c] for c in s)),
    ).reset_index()
    detail = pd.read_sql_query(
        "SELECT txn_id, voucher_no, source, posting_date, vendor_id, vendor_name, dept_name, "
        "account_name, requester_id, approver_id, invoice_no, amount FROM v_txn", conn)
    rep = agg.merge(detail, on="txn_id", how="left")
    rep = rep.sort_values(["risk_score", "amount"], ascending=[False, False]).reset_index(drop=True)
    rep.insert(0, "priority_rank", np.arange(1, len(rep) + 1))
    return rep
