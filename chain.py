"""Continue a finished ASPECT run from its checkpoint when its verdict is still open, so the next segment starts in the
cloud at once instead of after the local download (the Mac may sleep: 2026-11-21 the diag2 results waited 9 h).
Called by cloud/github/chain.yml when a run of run.yml completes, or by hand for tests.
Usage: python chain.py <unpacked results dir> <run title> <run id> [dry|smoke]
  dry: only print the decision; smoke: also continue smoke runs (to test the chain itself).
Continues only when it is not a smoke run, every ASPECT piece exited 0 (stopped at the end time or at the wall clock
with a checkpoint; a crash is never continued), the segment limit is not reached, and the verdict is open:
  - tools/aspect_brune.py (M1): tools/diag/brune_m1_eval.py says UNDECIDED; at most one continuation (title without
    "-s2"); new end time T_PAPER * TOL for 9.0 and 8.6 TN/m, T_FAIL for 8.0 (PARAM_AUDIT 11.15);
  - tools/aspect_rift2d.py: tools/diag/rift2d_eval.py says UNDECIDED; segments up to MAX_SEGMENTS; same variables
    and end time as the run (PARAM_AUDIT 11.17).
The next segment is named <title without -sN>-s<N+1> and reads the checkpoint of this run (run.yml input
restart_from)."""
import contextlib, glob, io, json, os, re, subprocess, sys
import brune_m1_eval as ev
import rift2d_eval as r2

MAX_SEGMENTS = 4                   # technical: 30 Myr in the deepest box takes about two 5.3 h segments (PARAM_AUDIT 11.17)

d, title, rid = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else ""
run = json.load(open(glob.glob(f"{d}/*_run.json")[0]))
rift2d = run.get("script") == "aspect_rift2d.py"
if run.get("script") not in (None, "aspect_rift2d.py"):   # aspect_brune.py runs carry no "script"; others have no chain
    print(f"not continued: no chain for {run['script']}")
    raise SystemExit
m = re.search(r"-s(\d+)$", title)
seg, base = (int(m.group(1)), title[:m.start()]) if m else (1, title)
why = ("no ASPECT run" if not run.get("pieces") else
       "smoke run" if run["smoke"] and mode != "smoke" else
       "an ASPECT piece failed" if any(p["exit"] != "0" for p in run["pieces"]) else
       "segment limit reached" if seg >= (MAX_SEGMENTS if rift2d else 2) else "")
if not why:
    if rift2d:
        v = r2.verdict(d)
        print(v)
        why = "verdict already decided" if v["verdict"] != "UNDECIDED" else ""
    else:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            ev.main([d])
        print(buf.getvalue())
        why = "verdict already decided" if "UNDECIDED" not in buf.getvalue() else ""
if why:
    print(f"not continued: {why}")
    raise SystemExit
if rift2d:
    script, args = "aspect_rift2d.py", [f"ENV:{k}={v}" for k, v in run["env"].items()]
else:
    F = run["F_TN_m"]
    end = (ev.T_PAPER[F] * ev.TOL if F in ev.T_PAPER else ev.T_FAIL)
    script, args = "aspect_brune.py", [f"ENV:BRUNE_F={F:g}", f"ENV:ASPECT_END_MYR={end:g}"]
args += ["ENV:ASPECT_SMOKE=1"] if run["smoke"] else []
cmd = ["gh", "workflow", "run", "run.yml", "-R", os.environ.get("GITHUB_REPOSITORY", "minuet/owg-compute"),
       "-f", f"script={script}", "-f", "args=" + json.dumps(args), "-f", f"name={base}-s{seg + 1}",
       "-f", f"restart_from={rid}"]
print("continue:", " ".join(cmd))
if mode != "dry":
    subprocess.run(cmd, check=True)
