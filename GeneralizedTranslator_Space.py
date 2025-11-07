import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal

def transmogrify(
        source_df: pd.DataFrame,
        source_id: str,
        target_id: str,
        weights: pd.DataFrame,
        val_cols: List[str],
        val_operators: Optional[Dict[str, Literal["sum", "mean", "rate"]]] = None,
        denom_cols: Optional[Dict[str, str]] = None
    ):
    w = weights[[source_id, target_id, "weight"]].copy()
    src = source_df[[source_id] + val_cols + ([] if denom_cols is None else list(denom_cols.values()))].copy()

    df = w.merge(src, on=source_id, how="left") # TODO: is this correct? check so that concat behaves as it should

    if val_operators is None:
        val_operators = {c: "sum" for c in val_cols}

    out_parts = []
    for col in val_cols:
        val_op = val_operators.get(col, "sum")
        if val_op in ("sum"):
            contrib = df[[target_id]].copy()
            contrib[col] = df[col] * df["weight"]
            part = contrib.groupby(target_id, dropna=False)[col].sum(min_count=1)
        elif val_op in ("mean", "rate"): #TODO: I need to think about this, does it actually work?
            if denom_cols and col in denom_cols:
                denom_col = denom_cols[col]
                num = (df[col] * df[denom_col] * df["weight"]).groupby(df[target_id], dropna=False).sum(min_count=1)
                den = (df[denom_col] * df["weight"]).groupby(df[target_id], dropna=False).sum(min_count=1)
                part = num / den
            else:
                num = (df[col] * df["weight"]).groupby(df[target_id], dropna=False).sum(min_count=1)
                den = (df["weight"]).groupby(df[target_id], dropna=False).sum(min_count=1)
                part = num / den
            part.name = col
        else:
            raise ValueError(f"Unknown value type for column '{col}': {val_operators}")
        out_parts.append(part)

    result = pd.concat(out_parts, axis=1)
    result.index.name = target_id
    return result.reset_index()

if __name__ == "__main__":
    # Example crosswalk (user-provided)
    weights = pd.DataFrame({
        "SRC": ["S1", "S1", "S2", "S3"],
        "TGT": ["T1", "T2", "T2", "T3"],
        "weight": [0.4, 0.6, 1.0, 1.0]
    })

    # Example source values
    source_values = pd.DataFrame({
        "SRC": ["S1", "S2", "S3"],
        "population": [100, 50, 75],
        "households": [40, 20, 25],
        "median_income": [30000, 40000, 35000],
    })

    # Define how to treat columns
    value_ops = {
        "population": "count",
        "households": "count",
        "median_income": "mean"
    }

    result = transmogrify(source_values, "SRC", "TGT", weights, ["population", "households", "median_income"], val_operators=value_ops)
    print(result)
