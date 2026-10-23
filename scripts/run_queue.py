#!/usr/bin/env python3
"""Run a list of shell commands one after another, logging each; skip none.
Usage: python scripts/run_queue.py queue.txt log_dir
A line starting with '#' is a comment. The queue file is read once at start."""
import os, subprocess, sys, time
queue, logdir = sys.argv[1], sys.argv[2]
os.makedirs(logdir, exist_ok=True)
cmds = [l.strip() for l in open(queue) if l.strip() and not l.startswith("#")]
for i, c in enumerate(cmds):
    name = f"{i:02d}_" + (c.split("--tag ")[1].split()[0] if "--tag " in c else "job")
    t0 = time.time()
    with open(os.path.join(logdir, name + ".log"), "w") as fh:
        fh.write(c + "\n\n"); fh.flush()
        rc = subprocess.call(c, shell=True, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} {(time.time()-t0)/60:.1f} min", flush=True)
print("QUEUE DONE", flush=True)
