"""Synthetic accounts-payable and purchasing-card ledger for a fictional university.

Generates roughly 420k transactions over three fiscal years (FY2023-FY2025,
July to June) totalling about $800M, together with master data (vendors,
departments, employees, GL accounts, university holidays) and a ground-truth
label file for nine planted anomaly types:

    duplicate_payment, split_purchase, weekend_holiday_posting, benford_deviation,
    round_dollar, vendor_amount_outlier, new_vendor_large_payment,
    self_approval, sequence_gap

A small number of data-quality defects (exact duplicate records, missing
approvers, blank amounts, unparseable dates, unknown vendor ids, untidy vendor
ids) are also injected so the ETL validation step has something to catch.
These defects are not anomalies and are not labelled.

Output (data/raw/):
    transactions_FY2023.csv, transactions_FY2024.csv, transactions_FY2025.csv
    vendors.csv, departments.csv, employees.csv, accounts.csv, holidays.csv
    anomaly_labels.csv   (ground truth, used only for evaluation)

Each transaction file is written per fiscal year to keep files under 20 MB.
The generator is deterministic (SEED below). To create a larger ledger, raise
N_BASE_TXNS (for example to 1_500_000); file sizes grow linearly, so you may
want to add more fiscal years or write monthly files instead.

Usage:
    python src/generate_data.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_BASE_TXNS = 418_000
TARGET_TOTAL = 800_000_000  # dollars, before planted anomalies
PERIOD_START = pd.Timestamp("2022-07-01")
PERIOD_END = pd.Timestamp("2025-06-30")
THRESHOLDS = (5_000, 25_000)  # secondary approval / competitive-bid thresholds

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def university_holidays(years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        rows += [
            (_observed(dt.date(y, 1, 1)), "New Year's Day"),
            (_nth_weekday(y, 1, 0, 3), "Martin Luther King Jr. Day"),
            (_last_weekday(y, 5, 0), "Memorial Day"),
            (_observed(dt.date(y, 6, 19)), "Juneteenth"),
            (_observed(dt.date(y, 7, 4)), "Independence Day"),
            (_nth_weekday(y, 9, 0, 1), "Labor Day"),
            (_nth_weekday(y, 11, 3, 4), "Thanksgiving Day"),
            (_nth_weekday(y, 11, 3, 4) + dt.timedelta(days=1), "Day after Thanksgiving"),
            (_observed(dt.date(y, 12, 24)) if dt.date(y, 12, 24).weekday() < 5
             else dt.date(y, 12, 23 if dt.date(y, 12, 24).weekday() == 5 else 22), "Christmas Eve"),
            (_observed(dt.date(y, 12, 25)), "Christmas Day"),
            (dt.date(y, 12, 31) if dt.date(y, 12, 31).weekday() < 5 else dt.date(y, 12, 29), "Winter Closure"),
        ]
    out = pd.DataFrame(rows, columns=["holiday_date", "holiday_name"])
    out["holiday_date"] = pd.to_datetime(out["holiday_date"])
    out = out.drop_duplicates("holiday_date")
    mask = out["holiday_date"].between(PERIOD_START - pd.Timedelta(days=60), PERIOD_END + pd.Timedelta(days=60))
    return out[mask].sort_values("holiday_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #

DEPARTMENTS = [
    ("Chemistry", "Arts & Sciences"), ("Physics", "Arts & Sciences"), ("Biology", "Arts & Sciences"),
    ("Mathematics", "Arts & Sciences"), ("History", "Arts & Sciences"), ("English", "Arts & Sciences"),
    ("Psychology", "Arts & Sciences"), ("Economics", "Arts & Sciences"),
    ("Mechanical Engineering", "Engineering"), ("Electrical Engineering", "Engineering"),
    ("Civil Engineering", "Engineering"), ("Computer Science", "Engineering"),
    ("Biomedical Engineering", "Engineering"), ("Nursing", "Health Sciences"),
    ("Public Health", "Health Sciences"), ("Pharmacy", "Health Sciences"),
    ("Medicine Research Labs", "Health Sciences"), ("Business School", "Business"),
    ("Law School", "Law"), ("Education", "Education"), ("University Library", "Academic Support"),
    ("Information Technology", "Administration"), ("Facilities Management", "Administration"),
    ("Campus Dining", "Auxiliary"), ("Housing & Residence Life", "Auxiliary"),
    ("Athletics", "Auxiliary"), ("Student Affairs", "Administration"),
    ("Admissions", "Administration"), ("Human Resources", "Administration"),
    ("Finance & Controller", "Administration"), ("Research Administration", "Administration"),
    ("Campus Police", "Administration"), ("Alumni Relations", "Advancement"),
    ("Communications & Marketing", "Advancement"), ("Graduate School", "Academic Support"),
]

ACCOUNTS = [
    ("51010", "Lab Supplies", "Supplies"), ("51020", "Office Supplies", "Supplies"),
    ("51030", "Computer Equipment < $5k", "Supplies"), ("51040", "Software Licences", "Services"),
    ("52010", "Professional Services", "Services"), ("52020", "Consulting", "Services"),
    ("52030", "Maintenance & Repairs", "Services"), ("52040", "Custodial Services", "Services"),
    ("52050", "Printing & Publications", "Services"), ("53010", "Utilities - Electricity", "Utilities"),
    ("53020", "Utilities - Gas & Water", "Utilities"), ("54010", "Travel - Airfare", "Travel"),
    ("54020", "Travel - Lodging", "Travel"), ("54030", "Meals & Hospitality", "Travel"),
    ("55010", "Capital Equipment", "Capital"), ("55020", "Building Improvements", "Capital"),
    ("56010", "Food Service Purchases", "Auxiliary"), ("56020", "Athletic Supplies", "Auxiliary"),
    ("57010", "Subscriptions & Memberships", "Services"), ("57020", "Event Services", "Services"),
    ("58010", "Student Stipends & Fellowships", "Student Support"),
    ("58020", "Sub-awards", "Sponsored Programs"), ("59010", "Freight & Shipping", "Supplies"),
    ("59020", "Rentals & Leases", "Services"),
]
STIPEND_ACCOUNT = "58010"
PCARD_ACCOUNTS = ["51010", "51020", "51030", "54010", "54020", "54030", "57010", "59010", "56020"]
AP_ACCOUNTS = [a[0] for a in ACCOUNTS if a[0] not in ("54030",)]

FIRST = ["Avery", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Parker", "Reese", "Rowan",
         "Elliot", "Hayden", "Skyler", "Dakota", "Emerson", "Finley", "Harper", "Kendall", "Logan", "Sage"]
LAST = ["Abbott", "Barnes", "Calloway", "Delgado", "Ellison", "Fairbanks", "Garrison", "Holloway",
        "Ingram", "Jennings", "Kendrick", "Lowell", "Merriman", "Norwood", "Oakes", "Prescott",
        "Quigley", "Radcliffe", "Sterling", "Thornton", "Underhill", "Vance", "Whitfield", "Yardley"]
V_PRE = ["Northfield", "Bluestone", "Cedar Ridge", "Summit", "Harbor", "Ironwood", "Lakeside", "Granite",
         "Silverline", "Riverbend", "Oak Valley", "Pinecrest", "Westgate", "Brightwater", "Redwood",
         "Keystone", "Meridian", "Crescent", "Highland", "Prairie", "Sterling", "Evergreen", "Fox Hollow",
         "Copperfield", "Atlas", "Beacon", "Compass", "Lumen", "Vantage", "Horizon"]
V_MID = ["Scientific", "Office", "Facility", "Technical", "Lab", "Campus", "Data", "Energy", "Print",
         "Medical", "Food", "Travel", "Building", "Network", "Safety", "Athletic", "Analytics", "Supply"]
V_SUF = ["Supply Co.", "Services LLC", "Solutions Inc.", "Group", "Partners", "Systems", "Distributors",
         "Corp.", "& Sons", "Associates"]
MERCHANTS = ["Campus Bookstore", "Airline", "Hotel", "Office Mart", "Lab Depot", "Cafe", "Shipping Hub",
             "Online Marketplace", "Electronics Outlet", "Conference Registration", "Sports Outfitters"]


def build_departments() -> pd.DataFrame:
    df = pd.DataFrame(DEPARTMENTS, columns=["dept_name", "college"])
    df.insert(0, "dept_id", [f"D{i:03d}" for i in range(1, len(df) + 1)])
    # department activity weight: auxiliary/admin/research units buy more
    w = rng.lognormal(0, 0.5, len(df))
    w[df["dept_name"].isin(["Facilities Management", "Information Technology", "Medicine Research Labs",
                            "Campus Dining", "Housing & Residence Life"]).to_numpy()] *= 3
    df["activity_weight"] = w / w.sum()
    return df


def build_employees(depts: pd.DataFrame) -> pd.DataFrame:
    rows, eid = [], 1
    for dept_id in depts["dept_id"]:
        n_staff = int(rng.integers(8, 16))
        roles = ["Department Head"] + ["Manager"] * 2 + ["Staff"] * n_staff
        for role in roles:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            rows.append((f"E{eid:04d}", name, dept_id, role))
            eid += 1
    return pd.DataFrame(rows, columns=["employee_id", "employee_name", "dept_id", "role"])


def build_vendors(n_ap: int = 2_600, n_pcard: int = 400, n_new_legit: int = 350, n_shell: int = 40):
    used = set()

    def name_ap():
        while True:
            n = f"{rng.choice(V_PRE)} {rng.choice(V_MID)} {rng.choice(V_SUF)}"
            if n not in used:
                used.add(n)
                return n
            n2 = f"{n} {int(rng.integers(2, 99))}"
            if n2 not in used:
                used.add(n2)
                return n2

    recs = []
    vid = 10001
    span_days = (PERIOD_END - PERIOD_START).days
    for kind, n in (("AP", n_ap), ("PCARD", n_pcard), ("AP_NEW", n_new_legit), ("SHELL", n_shell)):
        for _ in range(n):
            if kind == "PCARD":
                name = f"{rng.choice(MERCHANTS)} #{int(rng.integers(100, 9999))}"
                mu, sigma = rng.normal(np.log(110), 0.6), rng.uniform(0.45, 0.9)
                acct = rng.choice(PCARD_ACCOUNTS)
                created = PERIOD_START - pd.Timedelta(days=int(rng.integers(200, 3000)))
            else:
                name = name_ap()
                mu, sigma = rng.normal(np.log(1_400), 1.05), rng.uniform(0.35, 0.85)
                acct = rng.choice(AP_ACCOUNTS)
                if kind == "AP":
                    created = PERIOD_START - pd.Timedelta(days=int(rng.integers(30, 5000)))
                elif kind == "AP_NEW":
                    created = PERIOD_START + pd.Timedelta(days=int(rng.integers(0, span_days - 60)))
                else:  # shell vendor, created during the period
                    created = PERIOD_START + pd.Timedelta(days=int(rng.integers(60, span_days - 150)))
                    acct = rng.choice(["52010", "52020", "52030"])
            pattern = int(rng.integers(0, 4))
            recs.append((f"V{vid}", name, "PCARD" if kind == "PCARD" else "AP", acct, created,
                         mu, sigma, pattern, kind))
            vid += 1
    v = pd.DataFrame(recs, columns=["vendor_id", "vendor_name", "vendor_type", "primary_account",
                                    "created_date", "mu", "sigma", "inv_pattern", "_kind"])
    # Stipend payees are paid fixed round amounts (multiples of $500): a legitimate round-dollar population
    stip = v.index[(v["_kind"] == "AP")][:60]
    v.loc[stip, "primary_account"] = STIPEND_ACCOUNT
    v.loc[stip, "mu"] = np.log(2_500)
    v.loc[stip, "sigma"] = 0.45
    # popularity (Zipf-like) within each type
    pop = rng.pareto(1.3, len(v)) + 0.05
    pop = np.minimum(pop, np.quantile(pop, 0.99))  # no single vendor dominates the ledger
    pop[(v["_kind"] == "SHELL").to_numpy()] = 0  # shell vendors only appear through planted payments
    pop[(v["_kind"] == "AP_NEW").to_numpy()] *= 0.6
    v["popularity"] = pop
    return v


# --------------------------------------------------------------------------- #
# Transactions
# --------------------------------------------------------------------------- #

def business_day_roll(dates: pd.Series, holidays: set) -> pd.Series:
    d = dates.copy()
    for _ in range(6):
        bad = (d.dt.weekday >= 5) | d.isin(holidays)
        if not bad.any():
            break
        d[bad] = d[bad] + pd.Timedelta(days=1)
    return d


def assign_approvers(df: pd.DataFrame, emps: pd.DataFrame) -> pd.Series:
    heads = emps[emps.role == "Department Head"].set_index("dept_id")["employee_id"]
    mgr = emps[emps.role == "Manager"].groupby("dept_id")["employee_id"].apply(list)
    pick = rng.integers(0, 2, len(df))
    mgr_ids = np.array([mgr[d][i] for d, i in zip(df["dept_id"], pick)])
    head_ids = heads.reindex(df["dept_id"]).to_numpy()
    return pd.Series(np.where(df["amount"].abs() >= THRESHOLDS[0], head_ids, mgr_ids), index=df.index)


def generate_base(vendors, depts, emps, holidays):
    n = N_BASE_TXNS
    pcard_share = 0.45
    is_pcard = rng.random(n) < pcard_share
    vend = np.empty(n, dtype=object)
    for typ, mask in (("PCARD", is_pcard), ("AP", ~is_pcard)):
        pool = vendors[(vendors.vendor_type == typ) & (vendors.popularity > 0)]
        p = pool["popularity"].to_numpy() / pool["popularity"].sum()
        vend[mask] = rng.choice(pool["vendor_id"].to_numpy(), size=mask.sum(), p=p)
    df = pd.DataFrame({"vendor_id": vend, "source": np.where(is_pcard, "PCARD", "AP")})
    vi = vendors.set_index("vendor_id")
    df["dept_id"] = rng.choice(depts["dept_id"], size=n, p=depts["activity_weight"])
    # transaction / invoice date, respecting vendor creation date
    span = (PERIOD_END - PERIOD_START).days - 10
    created = vi.loc[df["vendor_id"], "created_date"].to_numpy()
    start = np.maximum(created, np.datetime64(PERIOD_START)).astype("datetime64[D]")
    end = np.datetime64(PERIOD_START + pd.Timedelta(days=span), "D")
    width = (end - start).astype(int)
    offs = (rng.random(n) * np.maximum(width, 1)).astype(int)
    inv_date = pd.Series(start + offs.astype("timedelta64[D]")).astype("datetime64[ns]")
    df["invoice_date"] = inv_date
    # amounts: vendor log-normal
    mu = vi.loc[df["vendor_id"], "mu"].to_numpy()
    sg = vi.loc[df["vendor_id"], "sigma"].to_numpy()
    df["amount"] = np.exp(rng.normal(mu, sg))
    df["account_code"] = vi.loc[df["vendor_id"], "primary_account"].to_numpy()
    # PCARD purchases cap at the card limit
    df.loc[is_pcard, "amount"] = df.loc[is_pcard, "amount"].clip(upper=4_999)
    # requesters are staff of the department
    staff = emps[emps.role == "Staff"].groupby("dept_id")["employee_id"].apply(np.array)
    # activity weights of staff (a few very active purchasers)
    act = {e: w for e, w in zip(emps["employee_id"], rng.lognormal(0, 0.6, len(emps)))}
    req = np.empty(n, dtype=object)
    for d, idx in df.groupby("dept_id").groups.items():
        cand = staff[d]
        w = np.array([act[e] for e in cand])
        req[np.asarray(idx)] = rng.choice(cand, size=len(idx), p=w / w.sum())
    df["requester_id"] = req
    # posting dates
    ap = df["source"] == "AP"
    lag = pd.to_timedelta(rng.integers(3, 30, n), unit="D")
    post = df["invoice_date"] + lag
    post_ap = business_day_roll(post[ap], holidays)
    post_pc = df.loc[~ap, "invoice_date"] + pd.to_timedelta(rng.integers(1, 3, (~ap).sum()), unit="D")
    df["posting_date"] = pd.concat([post_ap, post_pc]).sort_index()
    # Year-end close: AP batches posted on 30 June even when it falls on a weekend
    ye = ap & df["posting_date"].dt.month.eq(6) & df["posting_date"].dt.day.ge(26) & (rng.random(n) < 0.35)
    df.loc[ye, "posting_date"] = pd.to_datetime(df.loc[ye, "posting_date"].dt.year.astype(str) + "-06-30")
    # stipends: fixed round amounts
    st = df["account_code"] == STIPEND_ACCOUNT
    df.loc[st, "amount"] = (np.round(df.loc[st, "amount"] / 500) * 500).clip(lower=500)
    # credit memos
    cm = ap & (rng.random(n) < 0.01)
    df.loc[cm, "amount"] = -df.loc[cm, "amount"] * rng.uniform(0.05, 0.6, cm.sum())
    df["_label"] = ""
    return df


def scale_to_target(df):
    st = df["account_code"] == STIPEND_ACCOUNT
    total = df["amount"].sum()
    st_total = df.loc[st, "amount"].sum()
    factor = (TARGET_TOTAL - st_total) / (total - st_total)
    df.loc[~st, "amount"] = df.loc[~st, "amount"] * factor
    pc = df["source"] == "PCARD"
    df.loc[pc, "amount"] = df.loc[pc, "amount"].clip(upper=4_999)
    df["amount"] = df["amount"].round(2)
    return df


def plant_benford(df, emps, n_people: int = 6):
    """A handful of requesters whose amounts are fabricated (human-chosen digits)."""
    staff = emps[emps.role == "Staff"]["employee_id"].to_numpy()
    counts = df["requester_id"].value_counts()
    eligible = [e for e in staff if 150 <= counts.get(e, 0) <= 700]
    people = rng.choice(eligible, size=n_people, replace=False)
    m = df["requester_id"].isin(people)
    k = m.sum()
    first = rng.choice(np.arange(1, 10), size=k, p=np.array([2, 3, 5, 8, 12, 14, 18, 20, 18]) / 100)
    mag = rng.choice([10, 100, 1_000], size=k, p=[0.25, 0.5, 0.25])
    frac = rng.random(k)
    amt = (first + frac) * mag
    amt = np.where(rng.random(k) < 0.4, np.round(amt), np.round(amt, 2))
    pc = (df.loc[m, "source"] == "PCARD").to_numpy()
    amt = np.where(pc, np.minimum(amt, 4_999), amt)
    df.loc[m, "amount"] = amt
    df.loc[m, "_label"] = "benford_deviation"
    return df, list(people)


def plant_split_purchases(df, vendors, emps, holidays, n_groups: int = 260):
    ap_v = vendors[(vendors._kind == "AP") & (vendors.primary_account != STIPEND_ACCOUNT)]
    staff = emps[emps.role == "Staff"]
    rows = []
    for g in range(n_groups):
        T = THRESHOLDS[0] if rng.random() < 0.7 else THRESHOLDS[1]
        k = int(rng.choice([2, 3, 4], p=[0.55, 0.3, 0.15]))
        person = staff.iloc[int(rng.integers(len(staff)))]
        v = ap_v.iloc[int(rng.integers(len(ap_v)))]
        base = PERIOD_START + pd.Timedelta(days=int(rng.integers(30, 1_050)))
        for j in range(k):
            amt = round(float(rng.uniform(0.80, 0.995) * T), 2)
            d = base + pd.Timedelta(days=int(rng.integers(0, 5)))
            rows.append(dict(vendor_id=v.vendor_id, source="AP", dept_id=person.dept_id,
                             invoice_date=d, amount=amt, account_code=v.primary_account,
                             requester_id=person.employee_id, _label="split_purchase", _grp=g))
    sp = pd.DataFrame(rows)
    sp["posting_date"] = business_day_roll(sp["invoice_date"] + pd.Timedelta(days=int(rng.integers(3, 8))), holidays)
    return sp.drop(columns="_grp")


def plant_new_vendor_payments(vendors, emps, holidays):
    shells = vendors[vendors._kind == "SHELL"]
    staff = emps[emps.role == "Staff"]
    rows = []
    for _, v in shells.iterrows():
        person = staff.iloc[int(rng.integers(len(staff)))]
        for _ in range(int(rng.integers(1, 4))):
            late = rng.random() < 0.1  # some payments arrive after the 90-day window
            days = int(rng.integers(95, 140)) if late else int(rng.integers(5, 80))
            d = v.created_date + pd.Timedelta(days=days)
            rows.append(dict(vendor_id=v.vendor_id, source="AP", dept_id=person.dept_id,
                             invoice_date=d, amount=round(float(rng.uniform(28_000, 180_000)), 2),
                             account_code=v.primary_account, requester_id=person.employee_id,
                             _label="new_vendor_large_payment"))
    nv = pd.DataFrame(rows)
    nv["posting_date"] = business_day_roll(nv["invoice_date"] + pd.Timedelta(days=5), holidays)
    return nv


def assign_invoice_numbers(df, vendors):
    vi = vendors.set_index("vendor_id")
    df = df.sort_values(["vendor_id", "invoice_date"], kind="stable")
    vstart = pd.Series(rng.integers(1_000, 60_000, len(vi)), index=vi.index)
    start = vstart.loc[df["vendor_id"]].to_numpy()
    # vendors issue invoices to other customers too, so numbers increase in irregular steps
    step = pd.Series(rng.integers(1, 6, len(df)), index=df.index)
    num = start + step.groupby(df["vendor_id"]).cumsum().to_numpy()
    pat = vi.loc[df["vendor_id"], "inv_pattern"].to_numpy()
    fmt = np.select([pat == 0, pat == 1, pat == 2],
                    ["INV-" + pd.Series(num).astype(str).str.zfill(6),
                     pd.Series(num).astype(str),
                     "A" + pd.Series(num).astype(str).str.zfill(7)],
                    "SO" + pd.Series(num).astype(str))
    pc = (df["source"] == "PCARD").to_numpy()
    fmt = np.where(pc, "CARD-" + pd.Series(rng.integers(10**8, 10**9, len(df))).astype(str), fmt)
    df["invoice_no"] = fmt
    return df.sort_index()


def plant_in_place(df, holidays):
    """Self-approval, weekend/holiday posting, round-dollar and vendor outliers on disjoint AP rows."""
    clean_ap = df.index[(df["_label"] == "") & (df["source"] == "AP") & (df["amount"] > 0)
                        & (df["account_code"] != STIPEND_ACCOUNT)].to_numpy().copy()
    rng.shuffle(clean_ap)
    n_self, n_week, n_round, n_z = 300, 420, 520, 260
    i0 = 0
    sel = clean_ap[i0:i0 + n_self]; i0 += n_self
    df.loc[sel, "requester_id"] = df.loc[sel, "approver_id"]
    df.loc[sel, "_label"] = "self_approval"

    sel = clean_ap[i0:i0 + n_week]; i0 += n_week
    hol = sorted(holidays)
    for ix in sel:
        d = df.at[ix, "posting_date"]
        if rng.random() < 0.25:
            near = [h for h in hol if abs((h - d).days) < 40]
            new = near[int(rng.integers(len(near)))] if near else d
        else:
            new = d + pd.Timedelta(days=int((5 - d.weekday()) % 7 + rng.integers(0, 2)))
        df.at[ix, "posting_date"] = new
    df.loc[sel, "_label"] = "weekend_holiday_posting"

    sel = clean_ap[i0:i0 + n_round]; i0 += n_round
    df.loc[sel, "amount"] = (np.round(df.loc[sel, "amount"] / 1_000) * 1_000).clip(lower=1_000) \
        + rng.choice([0, 1_000, 2_000, 5_000], size=n_round, p=[0.55, 0.2, 0.15, 0.1])
    df.loc[sel, "_label"] = "round_dollar"

    # outliers only for vendors with history
    vc = df["vendor_id"].value_counts()
    cand = [ix for ix in clean_ap[i0:] if vc[df.at[ix, "vendor_id"]] >= 40][:n_z]
    return df, cand


def plant_outliers(df, cand, vendors):
    sig = vendors.set_index("vendor_id").loc[df.loc[cand, "vendor_id"], "sigma"].to_numpy()
    k = rng.uniform(3.0, 8.0, len(cand))
    mult = np.minimum(np.exp(k * sig), 60)
    df.loc[cand, "amount"] = (df.loc[cand, "amount"] * mult).round(2)
    df.loc[cand, "_label"] = "vendor_amount_outlier"
    return df


def _variant(inv: str) -> str:
    r = rng.random()
    if r < 0.35:
        return inv.replace("-", "")
    if r < 0.6:
        return inv.lower()
    if r < 0.8:
        return inv + " "
    # transposed digits: a true duplicate that a normalised-key match will miss
    s = list(inv)
    digits = [i for i, c in enumerate(s) if c.isdigit()]
    if len(digits) >= 2:
        a, b = digits[-1], digits[-2]
        s[a], s[b] = s[b], s[a]
    return "".join(s)


def plant_duplicates(df, holidays, n: int = 650):
    cand = df.index[(df["_label"] == "") & (df["source"] == "AP") & (df["amount"] > 50)].to_numpy()
    src = rng.choice(cand, size=n, replace=False)
    dup = df.loc[src].copy()
    dup["posting_date"] = business_day_roll(dup["posting_date"] + pd.to_timedelta(rng.integers(2, 25, n), unit="D"), holidays)
    vary = rng.random(n) < 0.45
    dup.loc[vary, "invoice_no"] = [_variant(s) for s in dup.loc[vary, "invoice_no"]]
    dup["_label"] = "duplicate_payment"
    return pd.concat([df, dup], ignore_index=True)


def fiscal_year(d: pd.Series) -> pd.Series:
    return d.dt.year + (d.dt.month >= 7).astype(int)


def number_and_gap(df, n_planted: int = 140, n_natural: int = 45):
    df = df[df["posting_date"].between(PERIOD_START, PERIOD_END)]
    df = df.sort_values(["posting_date", "vendor_id", "invoice_date"], kind="stable").reset_index(drop=True)
    df["txn_id"] = np.arange(1_000_001, 1_000_001 + len(df))
    df["voucher_no"] = ""
    ap = df["source"] == "AP"
    fy = fiscal_year(df["posting_date"])
    seq = df[ap].groupby(fy[ap]).cumcount() + 1
    df.loc[ap, "voucher_no"] = "AP" + (fy[ap] % 100).astype(str) + "-" + seq.astype(str).str.zfill(7)
    # remove vouchers: planted deletions (labelled on the following voucher) and natural cancellations
    ap_clean = df.index[ap & (df["_label"] == "")].to_numpy()
    pick = rng.choice(ap_clean, size=n_planted + n_natural, replace=False)
    planted, natural = pick[:n_planted], pick[n_planted:]
    ap_idx = df.index[ap].to_numpy()
    pos = {ix: p for p, ix in enumerate(ap_idx)}
    removed = set(pick.tolist())
    gap_labels = []
    for ix in planted:
        p = pos[ix] + 1
        while p < len(ap_idx) and ap_idx[p] in removed:
            p += 1
        if p < len(ap_idx):
            gap_labels.append(int(df.at[ap_idx[p], "txn_id"]))
    df = df.drop(index=pick).reset_index(drop=True)
    return df, gap_labels


def inject_dq_defects(df):
    """Data-quality problems for the ETL validation layer (not anomalies)."""
    out = df.copy()
    out["amount"] = out["amount"].map(lambda x: f"{x:.2f}")
    out["posting_date"] = out["posting_date"].dt.strftime("%Y-%m-%d")
    out["invoice_date"] = out["invoice_date"].dt.strftime("%Y-%m-%d")
    clean = out.index[(out["_label"] == "") & (out["source"] == "PCARD")].to_numpy()
    pick = rng.choice(clean, size=2_600, replace=False)
    a, b, c, d, e, f = pick[:80], pick[80:140], pick[140:200], pick[200:260], pick[260:600], pick[600:]
    out.loc[a, "approver_id"] = ""
    out.loc[b, "amount"] = ""
    out.loc[c, "posting_date"] = rng.choice(["2024-13-45", "N/A", "31/02/2024"], size=len(c))
    out.loc[d, "vendor_id"] = "V99999"
    out.loc[f, "vendor_id"] = [" " + v.lower() if i % 2 else v + " " for i, v in enumerate(out.loc[f, "vendor_id"])]
    dupes = out.loc[e].copy()  # exact duplicate extract records
    out = pd.concat([out, dupes]).sort_values("txn_id", kind="stable").reset_index(drop=True)
    return out


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for old in RAW_DIR.glob("transactions_FY*.csv"):
        old.unlink()
    hol = university_holidays(range(2022, 2026))
    hol_set = set(hol["holiday_date"])
    depts = build_departments()
    emps = build_employees(depts)
    vendors = build_vendors()

    df = generate_base(vendors, depts, emps, hol_set)
    df = scale_to_target(df)
    df, benford_people = plant_benford(df, emps)
    sp = plant_split_purchases(df, vendors, emps, hol_set)
    nv = plant_new_vendor_payments(vendors, emps, hol_set)
    df = pd.concat([df, sp, nv], ignore_index=True)
    df["approver_id"] = assign_approvers(df, emps)
    df = assign_invoice_numbers(df, vendors)
    df, outlier_cand = plant_in_place(df, hol_set)
    df = plant_outliers(df, outlier_cand, vendors)
    df = plant_duplicates(df, hol_set)
    df, gap_ids = number_and_gap(df)

    labels = df.loc[df["_label"] != "", ["txn_id", "_label"]].rename(columns={"_label": "anomaly_type"})
    labels = pd.concat([labels, pd.DataFrame({"txn_id": gap_ids, "anomaly_type": "sequence_gap"})])
    labels = labels.sort_values(["txn_id", "anomaly_type"]).reset_index(drop=True)

    cols = ["txn_id", "voucher_no", "source", "invoice_no", "invoice_date", "posting_date", "vendor_id",
            "dept_id", "account_code", "requester_id", "approver_id", "amount"]
    raw = inject_dq_defects(df[cols + ["_label"]])
    fy = fiscal_year(pd.to_datetime(df.set_index("txn_id").loc[raw["txn_id"], "posting_date"]).reset_index(drop=True))
    for y in sorted(fy.unique()):
        part = raw.loc[(fy == y).to_numpy(), cols]
        part.to_csv(RAW_DIR / f"transactions_FY{y}.csv", index=False)
        print(f"transactions_FY{y}.csv: {len(part):,} rows")

    vendors[["vendor_id", "vendor_name", "vendor_type", "primary_account", "created_date"]].assign(
        created_date=lambda x: x.created_date.dt.strftime("%Y-%m-%d")).to_csv(RAW_DIR / "vendors.csv", index=False)
    depts[["dept_id", "dept_name", "college"]].to_csv(RAW_DIR / "departments.csv", index=False)
    emps.to_csv(RAW_DIR / "employees.csv", index=False)
    pd.DataFrame(ACCOUNTS, columns=["account_code", "account_name", "account_group"]).to_csv(
        RAW_DIR / "accounts.csv", index=False)
    hol.assign(holiday_date=hol.holiday_date.dt.strftime("%Y-%m-%d")).to_csv(RAW_DIR / "holidays.csv", index=False)
    labels.to_csv(RAW_DIR / "anomaly_labels.csv", index=False)

    print(f"total transactions: {len(df):,}  total amount: ${df['amount'].sum():,.0f}")
    print(labels["anomaly_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
