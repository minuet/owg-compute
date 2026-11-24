"""Where the ASPECT build's apt-get step hangs on GitHub runners (2026-11-24: killtest-l2 hung 83 min there, killtest-l4
hit the 900 s limit twice; the same step took at most 251 s in 138 earlier cloud logs). Runs the same apt-get commands
as tools/aspect_rift2d.py, without -qq, printing every output line with its elapsed time, and a process snapshot
whenever apt prints nothing for QUIET_S. Output: stdout only (runner.py keeps it in NAME.log).
Usage (GitHub): PACK_EXTRA=tools/diag/gh_apt_probe.py cloud/gh_submit.sh NAME gh_apt_probe.py"""
import os, subprocess, threading, time

LIMIT_S = 900       # technical: NET_TIMEOUT_S of the ASPECT tools (PARAM_AUDIT 11.17)
QUIET_S = 120       # technical: snapshot after this long without output, over the 90 s p90 of the whole step (PARAM_AUDIT 11.17)
SUDO = "sudo " if os.geteuid() != 0 else ""
t0 = time.time()


def say(s):
    print(f"[{time.time() - t0:7.1f}] {s}", flush=True)


def snapshot():
    say(subprocess.run("ps -eo pid,ppid,etime,stat,cmd --forest | grep -E 'apt|dpkg|http|gpgv|unattended|timeout' "
                       "| grep -v grep; " + SUDO + "fuser -v /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock "
                       "/var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>&1",
                       shell=True, capture_output=True, text=True).stdout)


def step(cmd):
    say(f"$ {cmd}")
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = [time.time()]

    def watch():
        while p.poll() is None:
            time.sleep(10)
            if time.time() - last[0] > QUIET_S:
                say("quiet, snapshot:")
                snapshot()
                last[0] = time.time()
    threading.Thread(target=watch, daemon=True).start()
    for line in p.stdout:
        last[0] = time.time()
        say(line.rstrip())
    say(f"[exit {p.wait()}]")


say("before apt:")
snapshot()
step(f"{SUDO}timeout -k 30 {LIMIT_S} apt-get update")
step(f"{SUDO}DEBIAN_FRONTEND=noninteractive timeout -k 30 {LIMIT_S} apt-get install -y libdeal.ii-dev cmake ninja-build")
