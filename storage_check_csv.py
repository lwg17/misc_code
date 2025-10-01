import os, sys, csv
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path

# Usage: python storage_check.py <full_path> [<full_path2> ...]

if len(sys.argv) < 2:
    print("Usage: python storage_check.py <full_path> [<full_path2> ...]")
    sys.exit(1)

csv_file = Path.home() / "NN_train_storage.csv"
getcontext().prec = 28  # high precision for GiB

rows = []
for base in sys.argv[1:]:
    if not os.path.isdir(base):
        rows.append([base, os.path.basename(base.rstrip("/")), 0, "0.000000000"])
        continue

    for root, _, files in os.walk(base):
        ckpts = [f for f in files if f.endswith(".ckpt")]
        if not ckpts:
            continue
        total_bytes = sum(os.path.getsize(os.path.join(root, f)) for f in ckpts)
        gib = (Decimal(total_bytes) / Decimal(1024**3)).quantize(
            Decimal("0.000000001"), rounding=ROUND_HALF_UP
        )
        rows.append([root, os.path.basename(root.rstrip("/")), len(ckpts), f"{gib:.9f}"])

# Append if file exists, else write header
file_exists = csv_file.exists()
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["path", "folder", "num_ckpt", "size_gib"])
    writer.writerows(rows)

print(f"Appended {len(rows)} rows to {csv_file}")
