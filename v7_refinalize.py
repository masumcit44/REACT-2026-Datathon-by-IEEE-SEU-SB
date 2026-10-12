"""Wait for v7_strengthen.py to finish, then re-run stack -> select -> report."""
import os
import sys
import time
import subprocess

import v7_common as C

LOG = os.path.join(C.ROOT, "v7_strengthen_console.log")
STEPS = [["v7_stack.py", "--pool", "core", "--tag", "core"],
         ["v7_stack.py", "--pool", "wide", "--tag", "wide"],
         ["v7_select.py"],
         ["v7_diag_consistency.py"],
         ["v7_report.py"]]


def main():
    t0 = time.time()
    while time.time() - t0 < 10800:
        if os.path.exists(LOG):
            with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
                if "DONE_STRENGTHEN" in f.read():
                    print("[refinalize] strengthen finished", flush=True)
                    break
        time.sleep(20)
    else:
        print("[refinalize] timeout waiting for strengthen", flush=True)
    time.sleep(10)                     # release memory
    for cmd in STEPS:
        r = subprocess.run([sys.executable] + cmd, cwd=C.ROOT, capture_output=True, text=True)
        out = (r.stdout or "").strip().splitlines()
        for line in out[-8:]:
            print("   " + line, flush=True)
        if r.returncode != 0:
            print(f"[refinalize] !! {cmd} rc={r.returncode}", flush=True)
            print((r.stderr or "")[-1500:], flush=True)
    print("DONE_REFINALIZE", flush=True)


if __name__ == "__main__":
    main()