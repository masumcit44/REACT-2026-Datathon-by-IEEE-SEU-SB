"""V7 finalizer - waits for the member pool artifacts, then runs the stacking head and the
report writer. Runs unattended so the long member-training job can finish on its own.

Order of operations:
  1. wait until every required member has {fold2,fold3,test} predictions in v7_out/
  2. short settle delay (let the training process release its memory)
  3. v7_stack.py --pool wide      (stacking head + gated promotion + final submission)
  4. v7_report.py                 (V7_REPORT.md + v7_out/v7_run_log.txt)
"""
import os
import sys
import time
import subprocess

import v7_common as C

MEMBERS = ["hgb183", "lgb183", "graphlgb", "xgb183"]
KINDS = ("fold2", "fold3", "test")
POOL = "wide"


def member_ready(m):
    return all(os.path.exists(os.path.join(C.OUT, f"{m}_{k}.npy")) for k in KINDS)


def wait_for_members(timeout_s=7200, poll_s=20):
    t0 = time.time()
    last = -1
    while time.time() - t0 < timeout_s:
        done = [m for m in MEMBERS if member_ready(m)]
        if len(done) != last:
            print(f"[finalize] members ready {len(done)}/{len(MEMBERS)}: {done} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            last = len(done)
        if len(done) == len(MEMBERS):
            print("[finalize] all members present", flush=True)
            return True
        time.sleep(poll_s)
    print("[finalize] TIMEOUT waiting for members; continuing with what exists", flush=True)
    return False


def run(cmd):
    print(f"[finalize] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-25:]
    for line in tail:
        print("   " + line, flush=True)
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()[-15:]
        print(f"[finalize] !! exit {r.returncode}", flush=True)
        for line in err:
            print("   " + line, flush=True)
    return r.returncode


def main():
    wait_for_members()
    time.sleep(45)                       # let the trainer release memory
    rc1 = run([sys.executable, "v7_stack.py", "--pool", "core", "--tag", "core"])
    rc2 = run([sys.executable, "v7_stack.py", "--pool", "wide", "--tag", "wide"])
    rc3 = run([sys.executable, "v7_select.py"])
    rc4 = run([sys.executable, "v7_report.py"])
    print(f"[finalize] rc: core={rc1} wide={rc2} select={rc3} report={rc4}", flush=True)
    print("DONE_FINALIZE", flush=True)


if __name__ == "__main__":
    main()