"""Continue a finished run of tools/aspect_brune.py from its checkpoint when its verdict is still open, so the next
segment starts in the cloud at once instead of after the local download (the Mac may sleep: 2026-11-21 the diag2
results waited 9 h). Called by cloud/github/chain.yml when a run of run.yml completes, or by hand for tests.
Usage: python chain.py <unpacked results dir> <run title> <run id> [dry|smoke]
  dry: only print the decision; smoke: also continue smoke runs (to test the chain itself).
Continues only when it is not a smoke run, every ASPECT piece exited 0 (stopped at the end time or at the wall clock
with a checkpoint; a crash is never continued), the run is a first segment (title without "-s2": at most one automatic
continuation, so a stalled run cannot loop), and tools/diag/brune_m1_eval.py says UNDECIDED. The new end time is the
end of the decision window, T_PAPER * TOL for 9.0 and 8.6 TN/m and T_FAIL for 8.0 (brune_m1_eval.py, PARAM_AUDIT 11.15).
The next segment is named <title>-s2 and reads the checkpoint of this run (run.yml input restart_from)."""
import contextlib, glob, io, json, os, subprocess, sys
import brune_m1_eval as ev

d, title, rid = sys.argv[1:4]
mode = sys.argv[4] if len(sys.argv) > 4 else ""
run = json.load(open(glob.glob(f"{d}/*_run.json")[0]))
F = run["F_TN_m"]
with contextlib.redirect_stdout(io.StringIO()) as buf:
    ev.main([d])
print(buf.getvalue())
why = ("smoke run" if run["smoke"] and mode != "smoke" else
       "an ASPECT piece failed" if any(p["exit"] != "0" for p in run["pieces"]) else
       "already a continuation" if "-s2" in title else
       "verdict already decided" if "UNDECIDED" not in buf.getvalue() else "")
if why:
    print(f"not continued: {why}")
    raise SystemExit
end = (ev.T_PAPER[F] * ev.TOL if F in ev.T_PAPER else ev.T_FAIL)
args = [f"ENV:BRUNE_F={F:g}", f"ENV:ASPECT_END_MYR={end:g}"] + (["ENV:ASPECT_SMOKE=1"] if run["smoke"] else [])
cmd = ["gh", "workflow", "run", "run.yml", "-R", os.environ.get("GITHUB_REPOSITORY", "minuet/owg-compute"),
       "-f", "script=aspect_brune.py", "-f", "args=" + json.dumps(args), "-f", f"name={title}-s2",
       "-f", f"restart_from={rid}"]
print("continue:", " ".join(cmd))
if mode != "dry":
    subprocess.run(cmd, check=True)
