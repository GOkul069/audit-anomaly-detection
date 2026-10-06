"""Score check output against the generator's planted-anomaly labels."""

from __future__ import annotations

import pandas as pd


def precision_recall(results: dict[str, pd.DataFrame], labels: pd.DataFrame) -> pd.DataFrame:
    """Row-level precision and recall of each check against its own planted anomaly type.

    A flagged row counts as a true positive only if it carries the label the check is
    designed to find; rows that are other planted anomalies count as false positives.
    """
    rows = []
    for name, res in results.items():
        flagged = set(res["txn_id"].astype(int))
        truth = set(labels.loc[labels["anomaly_type"] == name, "txn_id"].astype(int))
        tp = len(flagged & truth)
        rows.append({
            "check": name,
            "flagged": len(flagged),
            "planted": len(truth),
            "true_positives": tp,
            "precision": tp / len(flagged) if flagged else float("nan"),
            "recall": tp / len(truth) if truth else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["f1"] = 2 * out.precision * out.recall / (out.precision + out.recall)
    return out
