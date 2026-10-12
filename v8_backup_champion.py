import os
import json
import hashlib
import shutil
import pandas as pd
import numpy as np

ROOT = "d:/Desktop/React datathon"
BACKUP = os.path.join(ROOT, "v8_out", "champion_backup")
OUT = os.path.join(ROOT, "v7_out")
SUBMIT = os.path.join(ROOT, "submission_file")

FILES_TO_BACKUP = [
    # submission
    os.path.join(SUBMIT, "v7_final_submission.csv"),
    # V7 test / fold predictions
    os.path.join(OUT, "v7_final_test_scores_wide.npy"),
    os.path.join(OUT, "hgb183_test.npy"),
    os.path.join(OUT, "hgb183_fold3.npy"),
    os.path.join(OUT, "hgb183_fold2.npy"),
    os.path.join(OUT, "lgb183_test.npy"),
    os.path.join(OUT, "lgb183_fold3.npy"),
    os.path.join(OUT, "lgb183_fold2.npy"),
    os.path.join(OUT, "xgb183_test.npy"),
    os.path.join(OUT, "xgb183_fold3.npy"),
    os.path.join(OUT, "xgb183_fold2.npy"),
    os.path.join(OUT, "graphlgb_test.npy"),
    os.path.join(OUT, "graphlgb_fold3.npy"),
    os.path.join(OUT, "graphlgb_fold2.npy"),
    os.path.join(OUT, "lgb183_gate_test.npy"),
    os.path.join(OUT, "lgb183_gate_fold3.npy"),
    os.path.join(OUT, "lgb183_gate_fold2.npy"),
    os.path.join(OUT, "xgb183_gate_test.npy"),
    os.path.join(OUT, "xgb183_gate_fold3.npy"),
    os.path.join(OUT, "xgb183_gate_fold2.npy"),
    os.path.join(OUT, "megacb_gate_test.npy"),
    os.path.join(OUT, "megacb_gate_fold3.npy"),
    os.path.join(OUT, "megacb_gate_fold2.npy"),
    os.path.join(OUT, "megalgb_gate_test.npy"),
    os.path.join(OUT, "megalgb_gate_fold3.npy"),
    os.path.join(OUT, "megalgb_gate_fold2.npy"),
    os.path.join(OUT, "hgb600_gate_test.npy"),
    os.path.join(OUT, "hgb600_gate_fold3.npy"),
    os.path.join(OUT, "hgb600_gate_fold2.npy"),
    # stack / selection artifacts
    os.path.join(OUT, "v7_stack_results_wide.json"),
    os.path.join(OUT, "v7_stack_results_core.json"),
    os.path.join(OUT, "v7_selection.json"),
    os.path.join(OUT, "v7_candidate_scores_wide.csv"),
    os.path.join(OUT, "v7_candidate_scores_core.csv"),
    os.path.join(OUT, "v7_candidate_scores.csv"),
    os.path.join(OUT, "v7_validation_results.json"),
    os.path.join(OUT, "v7_submission_check_wide.json"),
    os.path.join(OUT, "v7_submission_check_core.json"),
    os.path.join(OUT, "graph_v3_meta.json"),
    os.path.join(OUT, "graph_v3_test.npy"),
    os.path.join(OUT, "graph_v3_train.npy"),
    os.path.join(OUT, "hgb183_meta.json"),
    os.path.join(OUT, "xgb183_meta.json"),
    os.path.join(OUT, "lgb183_meta.json"),
    os.path.join(OUT, "graphlgb_meta.json"),
    os.path.join(OUT, "lgb183_meta.json"),
]

# Also include legacy blend artifacts referenced in v7_stack.py if present
LEGACY_PATHS = [
    os.path.join(ROOT, "xgb_hgb600_blend_fold2_pred.npy"),
    os.path.join(ROOT, "xgb_hgb600_blend_fold3_pred.npy"),
    os.path.join(ROOT, "xgb_hgb600_blend_test_pred.npy"),
]

for p in LEGACY_PATHS:
    if os.path.exists(p):
        FILES_TO_BACKUP.append(p)

os.makedirs(BACKUP, exist_ok=True)

manifest_rows = []

for orig in FILES_TO_BACKUP:
    if not os.path.exists(orig):
        print(f"[backup] SKIP (missing): {orig}")
        continue
    rel = os.path.relpath(orig, ROOT)
    backup_path = os.path.join(BACKUP, os.path.basename(orig))
    shutil.copyfile(orig, backup_path)
    sz = os.path.getsize(backup_path)

    sha = hashlib.sha256()
    with open(backup_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    sha256 = sha.hexdigest()

    row = {
        "original_path": orig,
        "backup_path": backup_path,
        "file_size_bytes": sz,
        "sha256": sha256,
        "array_shape": "",
        "prediction_mean": "",
        "prediction_std": "",
        "prediction_min": "",
        "prediction_max": "",
        "transaction_id_alignment": "",
    }

    if orig.endswith(".npy"):
        arr = np.load(backup_path)
        row["array_shape"] = list(arr.shape)
        if arr.ndim == 1 and arr.size > 0:
            row["prediction_mean"] = float(arr.mean())
            row["prediction_std"] = float(arr.std())
            row["prediction_min"] = float(arr.min())
            row["prediction_max"] = float(arr.max())

    if os.path.basename(orig) == "v7_final_submission.csv":
        df = pd.read_csv(backup_path)
        ss = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
        row["transaction_id_alignment"] = (
            "exact"
            if df["transaction_id"].tolist() == ss["transaction_id"].tolist()
            else "differs"
        )

    manifest_rows.append(row)
    print(f"[backup] OK: {rel} -> {os.path.basename(backup_path)} sha256={sha256[:16]}...")

manifest = pd.DataFrame(manifest_rows)
manifest_path = os.path.join(ROOT, "v8_out", "champion_backup_manifest.csv")
manifest.to_csv(manifest_path, index=False)
print("\n[manifest] saved:", manifest_path)
print("  files backed up:", len(manifest_rows))
print("  total size MB:", round(manifest["file_size_bytes"].sum() / 1e6, 2))
