-- Analytical views over the loaded ledger. Each v_chk_* view implements one rule-based
-- anomaly check and returns one row per flagged transaction. The two statistical checks
-- (Benford first-digit test and per-vendor robust z-score) are computed in Python in
-- src/checks.py on top of v_txn.

DROP VIEW IF EXISTS v_txn;
CREATE VIEW v_txn AS
SELECT t.txn_id, t.voucher_no, t.source, t.invoice_no, t.invoice_key, t.invoice_date,
       t.posting_date, t.fiscal_year, t.vendor_id, v.vendor_name, v.vendor_type,
       v.created_date AS vendor_created_date, t.dept_id, d.dept_name, t.account_code,
       a.account_name, t.requester_id, t.approver_id, t.amount,
       CAST(strftime('%w', t.posting_date) AS INTEGER) AS posting_weekday,
       h.holiday_name
FROM transactions t
LEFT JOIN vendors v      ON v.vendor_id = t.vendor_id
LEFT JOIN departments d  ON d.dept_id = t.dept_id
LEFT JOIN accounts a     ON a.account_code = t.account_code
LEFT JOIN holidays h     ON h.holiday_date = t.posting_date;

-- 1. Duplicate payments: same vendor, same amount, same normalised invoice number.
--    The first posting is treated as the original; later postings are flagged.
DROP VIEW IF EXISTS v_chk_duplicate_payments;
CREATE VIEW v_chk_duplicate_payments AS
WITH ranked AS (
    SELECT txn_id, vendor_id, amount, invoice_key, posting_date,
           ROW_NUMBER() OVER w AS rn,
           FIRST_VALUE(txn_id) OVER w AS original_txn_id,
           FIRST_VALUE(posting_date) OVER w AS original_posting_date
    FROM transactions
    WHERE source = 'AP' AND amount > 0
    WINDOW w AS (PARTITION BY vendor_id, amount, invoice_key ORDER BY posting_date, txn_id)
)
SELECT txn_id, original_txn_id, original_posting_date,
       CAST(julianday(posting_date) - julianday(original_posting_date) AS INTEGER) AS days_after_original
FROM ranked WHERE rn > 1;

-- 2. Split purchases: two or more payments by the same requester to the same vendor within
--    7 days, each between 75% and 100% of an approval threshold, that together exceed it.
DROP VIEW IF EXISTS v_chk_split_purchases;
CREATE VIEW v_chk_split_purchases AS
WITH thresholds(threshold) AS (VALUES (5000.0), (25000.0)),
near AS (
    SELECT t.txn_id, t.requester_id, t.vendor_id, t.invoice_date, t.amount, th.threshold
    FROM transactions t
    JOIN thresholds th ON t.amount >= 0.75 * th.threshold AND t.amount < th.threshold
    WHERE t.source = 'AP'
),
clusters AS (
    SELECT a.txn_id, a.threshold, COUNT(*) AS n_in_window, SUM(b.amount) AS window_total
    FROM near a
    JOIN near b
      ON b.requester_id = a.requester_id AND b.vendor_id = a.vendor_id
     AND b.threshold = a.threshold
     AND ABS(julianday(b.invoice_date) - julianday(a.invoice_date)) <= 7
    GROUP BY a.txn_id, a.threshold
)
SELECT txn_id, threshold, n_in_window, ROUND(window_total, 2) AS window_total
FROM clusters
WHERE n_in_window >= 2 AND window_total > threshold;

-- 3. Weekend / holiday postings of AP vouchers (card transactions legitimately settle any day).
DROP VIEW IF EXISTS v_chk_weekend_holiday;
CREATE VIEW v_chk_weekend_holiday AS
SELECT txn_id, posting_date,
       CASE WHEN holiday_name IS NOT NULL THEN holiday_name
            WHEN posting_weekday = 0 THEN 'Sunday' ELSE 'Saturday' END AS reason
FROM v_txn
WHERE source = 'AP' AND (posting_weekday IN (0, 6) OR holiday_name IS NOT NULL);

-- 5. Round-dollar amounts: exact multiples of $1,000 at or above $1,000. The account group is
--    returned so known fixed-amount populations (e.g. student stipends) can be excluded.
DROP VIEW IF EXISTS v_chk_round_dollar;
CREATE VIEW v_chk_round_dollar AS
SELECT t.txn_id, t.amount, t.account_code, a.account_group
FROM transactions t
LEFT JOIN accounts a ON a.account_code = t.account_code
WHERE t.amount >= 1000 AND t.amount = CAST(t.amount AS INTEGER) AND CAST(t.amount AS INTEGER) % 1000 = 0;

-- 7. Large payments (>= $25,000) to vendors created less than 90 days before posting.
DROP VIEW IF EXISTS v_chk_new_vendor_large;
CREATE VIEW v_chk_new_vendor_large AS
SELECT txn_id, vendor_id, vendor_created_date, amount,
       CAST(julianday(posting_date) - julianday(vendor_created_date) AS INTEGER) AS vendor_age_days
FROM v_txn
WHERE amount >= 25000
  AND julianday(posting_date) - julianday(vendor_created_date) BETWEEN 0 AND 90;

-- 8. Self-approval: requester and approver are the same person.
DROP VIEW IF EXISTS v_chk_self_approval;
CREATE VIEW v_chk_self_approval AS
SELECT txn_id, requester_id, approver_id, amount
FROM transactions
WHERE requester_id = approver_id;

-- 9. Gaps in the AP voucher sequence (per fiscal-year prefix). Quarantined rows still count
--    as issued vouchers, so ETL rejects do not create false gaps. The voucher after a gap is flagged.
DROP VIEW IF EXISTS v_chk_sequence_gaps;
CREATE VIEW v_chk_sequence_gaps AS
WITH issued AS (
    SELECT txn_id, voucher_no FROM transactions WHERE voucher_no <> ''
    UNION ALL
    SELECT NULL AS txn_id, voucher_no FROM rejected_transactions WHERE voucher_no <> ''
),
seq AS (
    SELECT txn_id, voucher_no,
           substr(voucher_no, 1, 4) AS prefix,
           CAST(substr(voucher_no, 6) AS INTEGER) AS n
    FROM issued
),
lagged AS (
    SELECT txn_id, voucher_no, prefix, n,
           LAG(n) OVER (PARTITION BY prefix ORDER BY n) AS prev_n
    FROM seq
)
SELECT txn_id, voucher_no, prev_n, n - prev_n - 1 AS missing_vouchers
FROM lagged
WHERE prev_n IS NOT NULL AND n - prev_n > 1 AND txn_id IS NOT NULL;
