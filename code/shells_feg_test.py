"""E-1119-16/-17, Kaggle: the authors' own Earth example ListEx1 (9 models of Earth5-049, their inferred tractions, 4 Shells
iterations each, the exact command recorded in its Models.txt) run on a copy of Earth5R.feg whose faults inside PB2002
plates are healed (tools/shells_feg_heal.py), so that only plate-boundary faults remain, as a generated planet's grid
would have. Everything else is the authors' input, unchanged; the unhealed run reproduced their table exactly
(tools/results/kaggle/shellset_earth/).
SHELLS_FEG_MODE chooses the grid:
  heal (default): Earth5R.feg with the faults inside plates healed (E-1119-16)
  gen: a grid generated from the PB2002 boundaries alone (tools/shells_grid.py, E-1119-17); its nodal data are 0, so
       OrbData must fill them: ListInput.in gets a third variable tSurf = 273 K for every model, which is the value of
       iEarth5-049.in, changes nothing, and makes ShellSet run OrbData (MOD_ShellSet.f90, OrbDataCheck)
  orbdata: the authors' Earth5R.feg with the same tSurf entry, i.e. its nodal data recomputed by ShellSet's OrbData from
       the grids shipped in INPUT/ (control for "gen": shows what the nodal-data source alone changes)
  SHELLS_GRID_REFINE=r (gen only): both grid spacings divided by r, for the convergence check; the array limits that
       OrbScore2.f90 and OrbData5.f90 fix at compile time are raised to the grid's own counts where these exceed them
  SHELLS_NODEDATA=own (gen only): nodal data written by orbdata_nodes.py (the port of OrbData, identical on Earth5R.feg)
       and no tSurf entry, so ShellSet does not run OrbData
SHELLS_HEAT=rule (gen only): node heat flow from shells_grid.heat_from_history (E-1120-05); =hf14: from HFgrid14 (needs
       HFgrid14.grd packed along, control of E-1120-06)
SHELLS_DIP_FLIP=OSR,CRB (gen only): these boundary classes get the reversed dip direction (E-1120-05)
SHELLS_DIP_RULE=elev (gen only): CCB and OCB faults dip toward the higher side (shells_grid.toward_higher, E-1120-06)
SHELLS_FAULT_FRIC=SUB:0.055,CCB:0.06 (gen only): these boundary classes get their own fault friction (shells_grid
       write_lr, rheologies 2, 3, ...; E-1121-03); the scanned fFric then applies to the other faults only
SHELLS_RESID=1 (gen only): basal drag = our slab flow + per-plate residual poles, the planet's method (E-1119-14, hcnoslab;
       PLATE_PLAN section 6 step 2), instead of the authors' 4 inversion iterations against PB2002 velocities; one step
SHELLS_TAG=x: appended to the output names, so variants of one mode can be told apart
SHELLS_SMOKE=1: the first model only, 1 iteration, 2 MPI ranks (a few minutes; checks that the grid is read and run).
Writes results/shells_feg_<mode>.json (grid counts, scores, the authors' reference scores) and results/shells_feg_<mode>_shellsF.zip
(torque reports, node velocities and logs of the final Shells runs)."""
import glob, json, os, re, time, zipfile
import shells_earth_flow as se
import shells_feg_heal as fh

SMOKE = os.environ.get("SHELLS_SMOKE") == "1"
MODE = os.environ.get("SHELLS_FEG_MODE", "heal")
REFINE = float(os.environ.get("SHELLS_GRID_REFINE", "1"))
OWN = os.environ.get("SHELLS_NODEDATA") == "own"
HEAT_RULE = {"rule": True, "hf14": "hf14"}.get(os.environ.get("SHELLS_HEAT"), False)
FLIP = tuple(c for c in os.environ.get("SHELLS_DIP_FLIP", "").split(",") if c)
DIP_ELEV = os.environ.get("SHELLS_DIP_RULE") == "elev"
WEAK = os.environ.get("SHELLS_WEAK_OROGENS", "")         # "all" or orogen names joined by ";" (E-1120-07)
WEAK = WEAK if WEAK in ("", "all") else WEAK.split(";")
SCAN = os.environ.get("SHELLS_SCAN", "")                  # "tauMax list;fFric list", comma-separated (E-1120-08)
FAULT_FRIC = [(c, float(v)) for c, v in (x.split(":") for x in os.environ.get("SHELLS_FAULT_FRIC", "").split(",") if x)]
TAG = os.environ.get("SHELLS_TAG", "")
STEPS = "PB2002_steps.dat"      # Bird 2003, shipped with the code (reference/earth/plate-boundaries/, cloud PACK_EXTRA)
S = se.S


def models_table(fn):
    """rows of Models.txt: model, ThID, the varied values (2, or 3 with tSurf), then GV SSR SD FSR SC SA GM; returned in
    the 11-column layout of shells_earth_flow.models_table (model, ThID, tauMax, fFric, 7 scores)"""
    rows = []
    for l in open(fn):
        f = l.split()
        if len(f) >= 11 and f[0].isdigit():
            v = [float(x) for x in f]
            rows.append(v[:4] + v[-7:])
    return rows


def main():
    se.LOG = open(se.RES / f"shells_feg_{MODE}{TAG}_steps.log", "w")
    summary = {"smoke": SMOKE, "mode": MODE}
    se.sh_run(f"git clone --depth 1 --branch v1.1.0 https://github.com/JonBMay/ShellSet.git {S}")
    if MODE == "heal":
        summary["heal"] = fh.heal(f"{S}/INPUT", f"{S}/healed")
        os.system(f"cp {S}/healed/* {S}/INPUT/")
    elif MODE == "gen":
        import shells_grid as sg
        summary["gen"] = g = sg.earth_grid(f"{S}/INPUT", STEPS, f"{S}/generated", REFINE, OWN, HEAT_RULE, FLIP, DIP_ELEV, WEAK,
                                              FAULT_FRIC)
        summary["refine"], summary["own_nodedata"], summary["heat_rule"], summary["flip"] = REFINE, OWN, HEAT_RULE, FLIP
        summary["dip_elev"], summary["weak_orogens"], summary["scan"] = DIP_ELEV, WEAK, SCAN
        summary["fault_fric"] = FAULT_FRIC
        summary["resid"] = se.RESID
        os.system(f"cp {S}/generated/* {S}/INPUT/")
        if se.RESID:
            _, summary["flow"], (out, vf, vplate) = se.hc_flow()
            flow_fn, summary["flow_noslab"] = se.write_noslab_flow(out, vf, vplate)
        need = {"maxNod": g["nodes"], "maxEl": g["triangles"], "maxFEl": g["faults"]}
        for fn in ("OrbScore2.f90", "OrbData5.f90"):     # compile-time array sizes, raised only where too small
            src = open(f"{S}/src/{fn}").read()
            for name, n in need.items():
                m = re.search(rf"(INTEGER, *PARAMETER *:: *{name} *= *)(\d+)", src)
                if m and int(m.group(2)) < n:
                    src = src.replace(m.group(0), f"{m.group(1)}{n}")
                    summary.setdefault("raised", []).append(f"{fn} {name} {m.group(2)} -> {n}")
            open(f"{S}/src/{fn}", "w").write(src)
    json.dump(summary, open(se.RES / f"shells_feg_{MODE}{TAG}.json", "w"), indent=1)
    summary["build_ok"] = se.shellset_build()
    if summary["build_ok"]:
        if se.RESID:   # parameter file with iConve = 2 and the flow file in the Main and Final sections, as E-1119-14
            se.setup_variant("hcnoslab", str(flow_fn))
        else:
            os.system(f"cp {S}/EXAMPLES/ListEx1/ListInput.in {S}/EXAMPLES/ListEx1/InputFiles.in {S}/INPUT/")
        if WEAK or FAULT_FRIC:    # third line of the "All:" block names the rheology table (MOD_ShellSet.f90 readSHELLS)
            fl = open(f"{S}/INPUT/InputFiles.in").read().splitlines()
            fl[3] = "LR_set.dat \t non-default Lithospheric Rheologies"
            open(f"{S}/INPUT/InputFiles.in", "w").write("\n".join(fl) + "\n")
        if SCAN:    # replace the authors' 3 x 3 tauMax x fFric list by our own (E-1120-08), same layout
            tau, ff = ([x for x in v.split(",")] for v in SCAN.split(";"))
            open(f"{S}/INPUT/ListInput.in", "w").write(
                f"2,{len(tau) * len(ff)}\ntauMax\nfFric\n---------- Var1:\n" + "".join(f"{t}\n" for t in tau for _ in ff)
                + "---------- Var2:\n" + "".join(f"{f}\n" for _ in tau for f in ff) + "----------\n")
        if SMOKE:   # same reduction as shells_earth_flow.setup_variant: keep the first of the 9 models
            li, out, n = open(f"{S}/INPUT/ListInput.in").read().splitlines(), [], None
            for l in li:
                if n is None and "," in l:
                    out.append(l.split(",")[0] + ",1"); n = 0
                elif l.startswith("----------"):
                    out.append(l); n = 0
                elif n is not None and l.strip() and l.strip()[0] in "0123456789.-" and len(l.split()) == 1:
                    if n == 0:
                        out.append(l)
                    n += 1
                else:
                    out.append(l)
            open(f"{S}/INPUT/ListInput.in", "w").write("\n".join(out) + "\n")
        if MODE == "orbdata" or (MODE == "gen" and not OWN):  # header "2,N", names, then one "---------- VarK:" block per variable
            li = open(f"{S}/INPUT/ListInput.in").read().splitlines()
            nv, nm = (int(x) for x in li[0].split(","))
            body = li[1:]
            k = body.index(next(l for l in body if l.startswith("----------")))
            body = body[:k] + ["tSurf"] + body[k:]
            if body[-1].strip() == "----------":
                body = body[:-1]
            body += [f"---------- Var{nv + 1}:"] + ["273."] * nm + ["----------"]
            open(f"{S}/INPUT/ListInput.in", "w").write(f"{nv + 1},{nm}\n" + "\n".join(body) + "\n")
        # ShellSet aborts with "more threads than ever needed" when ranks exceed models + 1 (rank 0 only distributes;
        # 4 ranks on 1 model failed on GitHub, 2 ranks on 1 model run): 4 ranks (the runner's cores) at most
        n_rank = min(2 if SMOKE else 4, int(open(f"{S}/INPUT/ListInput.in").readline().split(",")[1]) + 1)
        t = time.time()
        code, _ = se.sh_run(se.ONEAPI + "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; "
                            f"set -o pipefail; mpirun -np {n_rank} ./ShellSet.exe -Iter {1 if SMOKE or se.RESID else 4} "
                            "-InOpt List -Dir Lhealed -GM 5,SSR,GV,SD,SA,FSR 2>&1 | tail -60", cwd=S, timeout=se.RUN_TIMEOUT_S * 2)
        m = glob.glob(f"{S}/Lhealed/**/Models.txt", recursive=True)
        summary["healed"] = {"exit": code, "seconds": round(time.time() - t), "models": models_table(m[0]) if m else None}
        with zipfile.ZipFile(se.RES / f"shells_feg_{MODE}{TAG}_shellsF.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for f in glob.glob(f"{S}/Lhealed/ThID_*_ShellsF_output/*") + glob.glob(f"{S}/Lhealed/ThID_*_Data_output/*") + \
                    glob.glob(f"{S}/Lhealed/ThID_*_Shells_output/qEarth_*") + \
                    glob.glob(f"{S}/Error/*") + glob.glob(f"{S}/*Error*.txt") + m + [f"{S}/INPUT/Earth5R.feg", f"{S}/INPUT/ListInput.in"]:
                zf.write(f, os.path.relpath(f, S))
    summary["reference"] = se.models_table(f"{S}/EXAMPLES/ListEx1/Models.txt")
    json.dump(summary, open(se.RES / f"shells_feg_{MODE}{TAG}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
