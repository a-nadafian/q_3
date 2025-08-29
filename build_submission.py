import os
import glob
import zipfile
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


DATA_DIR = "/workspace/data"
OUTPUT_SUBMISSION = "/workspace/submission.csv"
OUTPUT_ZIP = "/workspace/result.zip"
NOTEBOOK_PATH = "/workspace/notebook.ipynb"
RANDOM_STATE = 42


def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lower_cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for key in lower_cols:
        if "time" in key or key in {"timestamp", "datetime"}:
            ts_col = lower_cols[key]
            break
    if ts_col is None:
        ts_col = df.columns[0]
    numeric_cols = [
        c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    val_col = numeric_cols[0] if numeric_cols else [c for c in df.columns if c != ts_col][0]
    return ts_col, val_col


def read_sensor_csv(path: str) -> Tuple[str, pd.DataFrame]:
    name = os.path.basename(path).replace("_test.csv", "")
    df = pd.read_csv(path)
    ts_col, val_col = infer_columns(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df[[ts_col, val_col]].rename(columns={ts_col: "timestamp", val_col: name})
    return name, df


def load_and_align(data_dir: str) -> Tuple[pd.DataFrame, List[str], str]:
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_test.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    sensor_to_df: Dict[str, pd.DataFrame] = {}
    sensor_lengths: Dict[str, int] = {}
    for f in csv_files:
        sensor_name, sdf = read_sensor_csv(f)
        sensor_to_df[sensor_name] = sdf
        sensor_lengths[sensor_name] = len(sdf)

    highest_sensor = max(sensor_lengths.items(), key=lambda kv: kv[1])[0]
    base_timeline = (
        sensor_to_df[highest_sensor][["timestamp"]]
        .dropna()
        .drop_duplicates()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    aligned = base_timeline.copy()
    for sensor_name, sdf in sensor_to_df.items():
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            base_timeline.sort_values("timestamp"),
            sdf,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("5min"),
        )
        aligned[sensor_name] = merged[sensor_name]

    aligned = aligned.sort_values("timestamp").reset_index(drop=True)
    value_cols = [c for c in aligned.columns if c != "timestamp"]
    aligned[value_cols] = aligned[value_cols].ffill().bfill()
    return aligned, value_cols, highest_sensor


def feature_engineer(aligned: pd.DataFrame, value_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    ROLL_WINDOWS = [5, 15, 60]
    NUM_LAGS = 3

    fe = aligned.copy()
    for col in value_cols:
        for w in ROLL_WINDOWS:
            r = fe[col].rolling(window=w, min_periods=max(2, w // 3))
            fe[f"{col}_rollmean_{w}"] = r.mean()
            fe[f"{col}_rollstd_{w}"] = r.std()
            fe[f"{col}_rollmin_{w}"] = r.min()
            fe[f"{col}_rollmax_{w}"] = r.max()

    for col in value_cols:
        for lag in range(1, NUM_LAGS + 1):
            fe[f"{col}_lag_{lag}"] = fe[col].shift(lag)

    feature_cols = [c for c in fe.columns if c != "timestamp"]
    fe[feature_cols] = fe[feature_cols].bfill().ffill()

    scaler = StandardScaler()
    X = scaler.fit_transform(fe[feature_cols])
    return fe, X, feature_cols


def train_and_predict(X: np.ndarray, fe: pd.DataFrame) -> pd.DataFrame:
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X)
    scores = iso.score_samples(X)
    labels = (iso.predict(X) == -1).astype(int)
    return pd.DataFrame(
        {
            "timestamp": fe["timestamp"].values,
            "anomaly_score": scores,
            "prediction": labels,
        }
    )


def save_submission(pred_df: pd.DataFrame, length_ref: int, out_path: str) -> None:
    submission = pd.DataFrame({"prediction": pred_df["prediction"].astype(int).values})
    if len(submission) != length_ref:
        raise ValueError("Submission length mismatch with highest-rate timeline")
    submission.to_csv(out_path, index=False)


def write_notebook(cells: List[str], notebook_path: str) -> None:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_code_cell(src) for src in cells]
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def main() -> None:
    np.random.seed(RANDOM_STATE)
    aligned, value_cols, highest = load_and_align(DATA_DIR)
    fe, X, feature_cols = feature_engineer(aligned, value_cols)
    pred_df = train_and_predict(X, fe)
    save_submission(pred_df, len(aligned), OUTPUT_SUBMISSION)

    code_cells = [
        """
# Setup: imports and constants
import os, glob, zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
DATA_DIR = \"/workspace/data\"
OUTPUT_SUBMISSION = \"/workspace/submission.csv\"
OUTPUT_ZIP = \"/workspace/result.zip\"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
""".strip(),
        """
# Load and align
from typing import Dict, Tuple
def infer_columns(df: pd.DataFrame):
    lower_cols = {c.lower(): c for c in df.columns}
    ts_col = None
    for key in lower_cols:
        if \"time\" in key or key in {\"timestamp\", \"datetime\"}:
            ts_col = lower_cols[key]; break
    if ts_col is None: ts_col = df.columns[0]
    numeric_cols = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    val_col = numeric_cols[0] if numeric_cols else [c for c in df.columns if c != ts_col][0]
    return ts_col, val_col
def read_sensor_csv(path: str):
    name = os.path.basename(path).replace(\"_test.csv\", \"\")
    df = pd.read_csv(path)
    ts_col, val_col = infer_columns(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors=\"coerce\")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    df = df[[ts_col, val_col]].rename(columns={ts_col: \"timestamp\", val_col: name})
    return name, df
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, \"*_test.csv\")))
sensor_to_df, sensor_lengths = {}, {}
for f in csv_files:
    s, sdf = read_sensor_csv(f); sensor_to_df[s]=sdf; sensor_lengths[s]=len(sdf)
highest_sensor = max(sensor_lengths.items(), key=lambda kv: kv[1])[0]
base_timeline = sensor_to_df[highest_sensor][[\"timestamp\"]].dropna().drop_duplicates().sort_values(\"timestamp\").reset_index(drop=True)
aligned = base_timeline.copy()
for s, sdf in sensor_to_df.items():
    sdf = sdf.sort_values(\"timestamp\").reset_index(drop=True)
    merged = pd.merge_asof(base_timeline.sort_values(\"timestamp\"), sdf, on=\"timestamp\", direction=\"nearest\", tolerance=pd.Timedelta(\"5min\"))
    aligned[s] = merged[s]
aligned = aligned.sort_values(\"timestamp\").reset_index(drop=True)
value_cols = [c for c in aligned.columns if c != \"timestamp\"]
aligned[value_cols] = aligned[value_cols].ffill().bfill()
print(f\"Loaded {len(csv_files)} sensors. Highest-rate sensor: {highest_sensor}. Timeline length: {len(aligned)}\")
""".strip(),
        """
# Feature engineering
ROLL_WINDOWS = [5, 15, 60]; NUM_LAGS = 3
fe = aligned.copy()
for col in value_cols:
    for w in ROLL_WINDOWS:
        r = fe[col].rolling(window=w, min_periods=max(2, w//3))
        fe[f\"{col}_rollmean_{w}\"] = r.mean(); fe[f\"{col}_rollstd_{w}\"] = r.std()
        fe[f\"{col}_rollmin_{w}\"] = r.min(); fe[f\"{col}_rollmax_{w}\"] = r.max()
for col in value_cols:
    for lag in range(1, NUM_LAGS+1):
        fe[f\"{col}_lag_{lag}\"] = fe[col].shift(lag)
feature_cols = [c for c in fe.columns if c != \"timestamp\"]
fe[feature_cols] = fe[feature_cols].bfill().ffill()
scaler = StandardScaler(); X = scaler.fit_transform(fe[feature_cols])
print(X.shape)
""".strip(),
        """
# Train and predict
iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(X)
scores = iso.score_samples(X)
labels = (iso.predict(X)==-1).astype(int)
pred_df = pd.DataFrame({\"timestamp\": fe[\"timestamp\"].values, \"anomaly_score\": scores, \"prediction\": labels})
pred_df.head()
""".strip(),
        """
# Save submission and zip
submission = pd.DataFrame({\"prediction\": pred_df[\"prediction\"].astype(int).values})
assert len(submission) == len(aligned)
submission.to_csv(OUTPUT_SUBMISSION, index=False)
with zipfile.ZipFile(OUTPUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(OUTPUT_SUBMISSION, arcname=os.path.basename(OUTPUT_SUBMISSION))
    zf.write(\"/workspace/notebook.ipynb\", arcname=\"notebook.ipynb\")
print(f\"Wrote {OUTPUT_SUBMISSION} and {OUTPUT_ZIP}\")
""".strip(),
    ]

    try:
        write_notebook(code_cells, NOTEBOOK_PATH)
    except Exception as e:
        # nbformat may be missing; defer writing notebook
        print(f"Warning: could not write notebook: {e}")

    # If notebook was created, ensure it's added to zip; otherwise zip only submission below in caller.


if __name__ == "__main__":
    main()

