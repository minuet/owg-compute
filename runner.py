"""GitHub Actions run of one tool script with arguments, the counterpart of cloud/tool_kernel.py on Kaggle.
RUN_SCRIPT, RUN_ARGS (JSON list; "ENV:NAME=value" items set variables) and RUN_NAME come from the workflow inputs.
Everything the tool writes to results/ that is small is zipped into out/NAME.zip, with its log."""
import glob, json, os, shutil, subprocess, sys, zipfile
SCRIPT, ARGS, NAME = os.environ["RUN_SCRIPT"], json.loads(os.environ["RUN_ARGS"]), os.environ["RUN_NAME"]
REPO = os.path.dirname(os.path.abspath(__file__))
C, R, O = "/tmp/w/code", "/tmp/w/results", f"{REPO}/out"
shutil.copytree(f"{REPO}/code", C); os.makedirs(R, exist_ok=True); os.makedirs(O, exist_ok=True)
for j in glob.glob(C + "/*.json") + glob.glob(C + "/*.npz") + glob.glob(f"{REPO}/earth/*.npz"):
    shutil.copy(j, R) if j.startswith(REPO + "/earth") else shutil.move(j, R)
os.symlink(R, C + "/results")
M = "/tmp/w/reference/earth/plate-model"; os.makedirs(M, exist_ok=True)
zipfile.ZipFile(f"{REPO}/earth/earth.zip").extractall(M)
before = set(glob.glob(R + "/*"))
stamp = open(C + "/code_stamp.txt").read().strip() if os.path.exists(C + "/code_stamp.txt") else "none"
with open(f"{R}/{NAME}.log", "w") as f:
    f.write(f"code stamp: {stamp}\n"); f.flush()
    env = dict(os.environ, MPLCONFIGDIR="/tmp/mpl")
    env.update(dict(a[4:].split("=", 1) for a in ARGS if a.startswith("ENV:")))
    code = subprocess.run([sys.executable, SCRIPT] + [a for a in ARGS if not a.startswith("ENV:")], cwd=C, env=env,
                          stdout=f, stderr=subprocess.STDOUT).returncode
files = [fn for fn in set(glob.glob(R + "/*")) - before | {f"{R}/{NAME}.log"} if os.path.isfile(fn)]
big = [fn for fn in files if os.path.getsize(fn) >= 2e8]
with open(f"{R}/{NAME}.log", "a") as f:     # say so in the downloaded log instead of dropping files silently
    for fn in big:
        f.write(f"\nNOT PACKED (too large): {os.path.basename(fn)} {os.path.getsize(fn) / 1e6:.0f} MB\n")
with zipfile.ZipFile(f"{O}/{NAME}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fn in files:
        if fn not in big:
            z.write(fn, os.path.basename(fn))
print(open(f"{R}/{NAME}.log").read()[-3000:])
sys.exit(code)
