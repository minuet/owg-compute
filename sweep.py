"""Cron sweep (cloud/github/sweep.yml): run chain.py on chained runs that chain.yml never saw.
GitHub starts no workflow_run event for a run dispatched with the workflow's own token, so chain.yml continues only the
first segment of a run; the sweep looks at every run of run.yml with a chained prefix that completed between GUARD_MIN
and WINDOW_H ago and has no successor (<base>-s<N+1>) yet, and lets chain.py decide.
Usage: python sweep.py [dry]"""
import datetime as dt, json, os, re, subprocess, sys

PREFIXES = ("brune-", "r2d-", "si2dc-")   # as chain.yml
WINDOW_H = 6       # technical: covers late cron starts (GitHub delays them up to about 1 h) several times (PARAM_AUDIT 11.20)
GUARD_MIN = 20     # technical: chain.yml needs about 1 min after a run ends; never continue a run twice (PARAM_AUDIT 11.20)

repo = os.environ["GITHUB_REPOSITORY"]
mode = sys.argv[1:2]
runs = json.loads(subprocess.run(["gh", "run", "list", "-R", repo, "-w", "run.yml", "-L", "300", "--json",
                                  "databaseId,displayTitle,status,updatedAt"],
                                 capture_output=True, text=True, check=True).stdout)
titles = {r["displayTitle"] for r in runs}
now = dt.datetime.now(dt.timezone.utc)
for r in runs:
    t, rid = r["displayTitle"], str(r["databaseId"])
    age = now - dt.datetime.fromisoformat(r["updatedAt"].replace("Z", "+00:00"))
    if (r["status"] != "completed" or not t.startswith(PREFIXES)
            or not dt.timedelta(minutes=GUARD_MIN) <= age <= dt.timedelta(hours=WINDOW_H)):
        continue
    m = re.search(r"-s(\d+)$", t)
    if (f"{t[:m.start()]}-s{int(m.group(1)) + 1}" if m else f"{t}-s2") in titles:
        continue
    print("==", t, rid, flush=True)
    d = f"sweep/{rid}"
    if subprocess.run(f'gh run download {rid} -R {repo} -n "{t}" -D {d}/dl && unzip -oq "{d}/dl/{t}.zip" -d {d}/res',
                      shell=True).returncode == 0:
        subprocess.run([sys.executable, "chain.py", f"{d}/res", t, rid] + mode)
    subprocess.run(["rm", "-rf", d])
