import lmdb, os, sys
p = sys.argv[1] if len(sys.argv)>1 else "data.mdb"
if os.path.isdir(p): p = os.path.join(p, "data.mdb")
if not os.path.exists(p): raise SystemExit(f"Error: {p} not found")
env = lmdb.open(os.path.dirname(p), readonly=True, lock=False)
with env.begin() as txn: entries = txn.stat()["entries"]
size_gib = os.path.getsize(p)/(1024**3)
print(f"LMDB file: {p}")
print(f"Entries  : {entries}")
print(f"Size     : {size_gib:.12f} GiB")
