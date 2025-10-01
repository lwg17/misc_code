import os, sys

if len(sys.argv) < 2:
    print("Usage: python storage_check.py <full_path> [<full_path2> ...]")
    sys.exit(1)

for base in sys.argv[1:]:
    if not os.path.isdir(base):
        print(f"Error: {base} is not a directory")
        continue

    grand_total_bytes = 0
    print(f"\nScanning: {base}")
    print("-" * 60)

    for root, dirs, files in os.walk(base):
        ckpts = [f for f in files if f.endswith(".ckpt")]
        if ckpts:
            total_bytes = sum(os.path.getsize(os.path.join(root, f)) for f in ckpts)
            total_gib = total_bytes / (1024**3)
            grand_total_bytes += total_bytes
            print(f"{root}: {len(ckpts)} ckpt files, "
                  f"{total_bytes} bytes = {total_gib:.9f} GiB")

    grand_total_gib = grand_total_bytes / (1024**3)
    print("=" * 60)
    print(f"Grand total across all checkpoints under {base}: "
          f"{grand_total_bytes} bytes = {grand_total_gib:.9f} GiB")
