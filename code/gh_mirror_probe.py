"""Download speed of Ubuntu mirrors from a GitHub runner (2026-11-24: apt-probe-1 got 48.7 MB in 891 s, about 55 kB/s,
from the runner's default us-east-2.ec2.archive.ubuntu.com, so libdeal.ii-dev's 424 MB could not finish in NET_TIMEOUT_S).
Each mirror serves the same file (dists/noble/Contents-amd64.gz); curl reports its mean speed over at most MAX_S.
Usage (GitHub): PACK_EXTRA=tools/diag/gh_mirror_probe.py cloud/gh_submit.sh NAME gh_mirror_probe.py"""
import subprocess

MAX_S = 20          # technical: long enough to pass TCP slow start at tens of MB/s, short enough for 2 rounds of 7 mirrors in 5 min
MIRRORS = ["http://us-east-2.ec2.archive.ubuntu.com/ubuntu", "http://us-east-1.ec2.archive.ubuntu.com/ubuntu",
           "http://azure.archive.ubuntu.com/ubuntu", "http://archive.ubuntu.com/ubuntu", "http://security.ubuntu.com/ubuntu",
           "http://mirrors.edge.kernel.org/ubuntu", "http://mirror.math.princeton.edu/pub/ubuntu"]
print(open("/etc/apt/apt-mirrors.txt").read(), flush=True)
for rnd in (1, 2):
    for m in MIRRORS:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "--max-time", str(MAX_S), "-w", "%{http_code} %{speed_download} %{size_download}",
                            f"{m}/dists/noble/Contents-amd64.gz"], capture_output=True, text=True).stdout.split()
        print(f"round {rnd} {m}: http {r[0]}, {float(r[1]) / 1e6:.2f} MB/s, {float(r[2]) / 1e6:.1f} MB", flush=True)
