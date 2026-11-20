"""3D globe preview page (rotatable, climate-coloured) for generated planets, for the user to judge the look (RULES.md B9b).

Usage: .venv/bin/python planet-history-sim/tools/planet_preview.py <out.html> seed [seed ...]
Uses the fine replay if present (islands visible), else the coarse run. Template: preview_template.html."""
import json, os, re, sys, base64, io, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from planet_screen import screen, to_grid, R_PLANET, LAT
from climate_rough import climate_grid
from plate_checks import HERE
COL = np.array([[22,110,60],[196,180,90],[226,196,140],[170,190,110],[80,150,80],[60,105,95],[150,160,140],[245,248,252],[200,200,210]], float)/255
W, H = 2048, 1024
lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H); lon = np.radians(-180 + (np.arange(W) + 0.5) * 360 / W)
la, lo = np.meshgrid(lat, lon, indexing="ij")
G = np.c_[(np.cos(la) * np.cos(lo)).ravel(), (np.cos(la) * np.sin(lo)).ravel(), np.sin(la).ravel()]
def b64(arr, **kw):
    buf = io.BytesIO(); plt.imsave(buf, arr, format="jpeg", pil_kwargs={"quality": 88}, **kw); return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
items = []
for rank, s in enumerate([int(a) for a in sys.argv[2:]], 1):
    src = HERE / f"results/planet_m1_seed{s}"
    fine = (src / "fine/terrain_fine.npz").exists()
    r = screen(s)
    t = np.load(src / "terrain.npz")
    e1 = to_grid(t["grid"], t["elev"]); B1 = climate_grid(e1, e1 > 0, R_PLANET)["B"]
    d = np.load(src / "fine/terrain_fine.npz") if fine else t
    elev = d["elev"][cKDTree(d["grid"]).query(G)[1]].reshape(H, W).astype(float)
    land = elev > 0
    # smooth biome boundaries: bilinear upsampling of each biome's 1-degree share, then the strongest one wins
    # (nearest-cell upsampling drew 1-degree stair steps); sea cells take the nearest land biome first
    i0, j0 = ndimage.distance_transform_edt(B1 < 0, return_distances=False, return_indices=True)
    B1f = B1[i0, j0]
    oh = np.stack([(B1f == k).astype(float) for k in range(9)])
    oh = np.stack([ndimage.zoom(np.pad(o, ((0, 0), (1, 1)), mode="wrap"), (H / 180, (W + 2 * W / 360) / 362), order=1)[:H, :W] for o in oh])
    B = np.argmax(oh, axis=0)
    B = np.where(B1[(np.arange(H) * 180 // H)[:, None], (np.arange(W) * 360 // W)[None, :]] >= 0, B, -1)
    dist, idx = ndimage.distance_transform_edt(B < 0, return_distances=True, return_indices=True)
    Bn = B[idx[0], idx[1]]
    # remote islands (> ~300 km from any climate land cell): climate by latitude instead of the nearest continent
    alat = np.abs(np.degrees(lat))[:, None] * np.ones((1, W))
    Blat = np.select([alat < 15, alat < 30, alat < 45, alat < 58, alat < 68], [0, 1, 4, 5, 6], 7)
    B = np.where(B >= 0, B, np.where(dist * (20000 / H) > 300, Blat, Bn))
    img = np.empty((H, W, 3))
    # sea floor drawn from smoothed depth (~60 km): the 22 km grid shows arc ridges as rows of beads otherwise
    sea = ndimage.gaussian_filter(np.where(land, -200.0, elev), 3.0, mode="wrap")
    dd = np.clip(-sea / 6000, 0, 1)[..., None] ** 0.7
    img[:] = (1 - dd) * np.array([95, 160, 205]) / 255 + dd * np.array([14, 38, 85]) / 255
    img[land] = COL[np.clip(B[land], 0, 8)]
    for c in range(3): img[..., c] = ndimage.gaussian_filter(img[..., c], 0.7, mode="wrap")
    gy, gx = np.gradient(np.where(land, elev, 0))
    img = np.where(land[..., None], np.clip(img * np.clip(1 - (gx - gy) / 700, 0.7, 1.2)[..., None], 0, 1), img)
    if os.environ.get("MARGINS") == "1" and (src / "records.npz").exists():
        # rifted margins, only rifts whose two sides drifted apart (user 2026-10-29: same-colour blobs on both sides of a barely
        # opened rift read as one continent cut by a knife): per rift, the coastal rifted crust splits into pieces on the present
        # map; the two largest pieces are drawn in that rift's own colour when they are now >= MIN_SEP km apart
        MIN_SEP = 800
        rt = np.round(np.load(src / "records.npz")["rift_time"]); gx_ = t["grid"]
        tree = cKDTree(gx_); sea_ = t["elev"] <= 0
        coastal = np.array([sea_[nb].any() for nb in tree.query_ball_point(gx_, 200 / R_PLANET)]) & ~sea_
        pal = np.array([[255, 40, 40], [255, 0, 255], [255, 150, 0], [0, 235, 255], [255, 255, 0], [255, 255, 255]]) / 255
        k = 0; mg_info = []
        # only rifts whose two sides still fit (conjugate_check: a FIT pair of major continents, each >= 3% of the land),
        # so a colour means "these two coasts fit" (user 2026-10-31: colouring every rift pair hid which ones fit)
        fit_T = None
        if (src / "conjugates.json").exists():
            fit_T = {round(p_["rift_myr"]) for p_ in json.loads((src / "conjugates.json").read_text())["pairs"]
                     if p_["fits"] and min(p_["landA"], p_["landB"]) >= 3}
        for T in sorted(set(rt[rt > 0].tolist())):
            if fit_T is not None and round(T) not in fit_T: continue
            q = gx_[(rt == T) & coastal]
            if len(q) < 6: continue
            sub = cKDTree(q); m = sub.sparse_distance_matrix(sub, 250 / R_PLANET).tocsr()

            nc, cl = connected_components(m, directed=False)
            # several rifts can share one 10-Myr time step, so pieces are paired only when their distance is reachable by
            # spreading since T (<= 100 km/Myr, i.e. 10 cm/yr full rate); only the biggest pair per rift, one colour each
            sizes = np.bincount(cl); pcs = [q[cl == i] for i in range(nc) if sizes[i] >= 3]
            cand = []
            for i in range(len(pcs)):
                for j in range(i + 1, len(pcs)):
                    sep = cKDTree(pcs[i]).query(pcs[j])[0].min() * R_PLANET
                    if MIN_SEP <= sep <= 100 * T: cand.append((min(len(pcs[i]), len(pcs[j])), i, j, sep))
            used = set()
            for _, i, j, sep in sorted(cand, reverse=True):
                if used: break
                used |= {i, j}; mg_info.append((T, round(sep)))
                for side in (pcs[i], pcs[j]):
                    yy = ((90 - np.degrees(np.arcsin(side[:, 2]))) / 180 * H).astype(int); xx = ((np.degrees(np.arctan2(side[:, 1], side[:, 0])) + 180) / 360 * W).astype(int)
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if dx * dx + dy * dy <= 4:
                                img[np.clip(yy + dy, 0, H - 1), (xx + dx) % W] = pal[k % len(pal)]
                k += 1
        print(s, "drifted rift pairs (Myr ago, km apart now):", mg_info)
    # landmasses and islands at this resolution
    cell = (np.pi * R_PLANET / H) ** 2 * np.cos(lat)[:, None] * np.ones((1, W))
    lab, n = ndimage.label(land)
    for y in range(H):
        a_, b_ = lab[y, 0], lab[y, -1]
        if a_ and b_ and a_ != b_: lab[lab == b_] = a_
    ids = np.unique(lab[lab > 0]); areas = ndimage.sum(cell, lab, ids); tot = areas.sum()
    big = sorted([a / tot for a in areas if a / tot >= 0.03], reverse=True)
    polar = float(cell[land & (np.abs(np.degrees(lat))[:, None] > 60)].sum() / cell[land].sum())
    st = r["story_starts"][0] if r["story_starts"] else [0, 0, 0, "none"]
    lk = json.loads((src / "look.json").read_text()) if (src / "look.json").exists() else None
    if lk:                                       # continents as look_check counts them (close straits merged, B9b)
        big = lk["continents"]
    mg = json.loads((src / "fine/migration.json").read_text()) if (src / "fine/migration.json").exists() else None
    items.append(dict(seed=s, rank=rank, strict=False, preview=True, hab=round(r["habitable"]*100,1), des=round(r["desert"]*100,1), cold=round(r["cold"]*100,1),
        ice=round(r["shares"][7]*100,1), tun=round(r["shares"][6]*100,1), rift=r["rift_start"], riftkm=None, start=[st[1], st[2]],
        big=[round(b*100) for b in big], polar=round(polar*100), walk=mg and round(mg["walk_or_raft_150km"]*100), boat=mg and round(mg["coastal_boat_1200km"]*100), finedata=fine,
        islands=int(np.sum((areas >= 1000) & (areas < 0.01 * tot))), tex=b64(img),
        bump=b64(np.clip(np.where(land, 0.25 + elev / 5000 * 0.75, 0.25 * (1 + elev / 7000)), 0, 1), cmap="gray", vmin=0, vmax=1)))
    print(s, "fine" if fine else "coarse", "landmasses", [round(b*100) for b in big], "islands", items[-1]["islands"])
page = open(HERE / "preview_template.html").read()
page = re.sub(r"const P = \[.*?\];\n", lambda m: "const P = " + json.dumps(items) + ";\n", page, count=1, flags=re.S)
page = page.replace("<title>候选星球地球仪</title>", "<title>新星球预览</title>").replace("<h1>候选星球地球仪</h1>", "<h1>新星球预览</h1>")
page = re.sub(r'<p class="sub">12 颗候选，按离目标的远近排序。', '<p class="sub">按新方法生成的预览，还没做科学检查和筛选，先看样子。', page)
page = page.replace('tags.push(p.strict ?', 'if (p.preview) tags.push(`<span class="tag">预览：还没做科学检查</span>`); else tags.push(p.strict ?')
page = page.replace('tags.push(p.rift ? `<span class="tag rift">起点旁有裂谷（约 ${p.riftkm} km）</span>` : `<span class="tag">起点靠高山</span>`);',
                    'tags.push(p.preview ? `<span class="tag">岛屿（≥ 1000 km²）${p.islands} 个</span>` : (p.rift ? `<span class="tag rift">起点旁有裂谷（约 ${p.riftkm} km）</span>` : `<span class="tag">起点靠高山</span>`));')
page = page.replace("<span>第 ${p.rank} 名${p.walk != null ? \" · 已细查\" : \"\"}</span>", "<span>${p.preview ? \"预览\" : \"第 \" + p.rank + \" 名\"}</span>")
page = page.replace('<h2>星球 ${p.seed} <span class="hint">第 ${p.rank} 名</span></h2>', '<h2>星球 ${p.seed}</h2>')
page = re.sub(r'<p class="note"><b>已知问题：</b>.*?</p>', '<p class="note">陆地按粗略气候着色，地形起伏放大了一些。红色标记是可能的故事起点（雨林挨着草原的地方）。这是预览：先看大陆、海岸和岛屿像不像一颗真的星球。' + ('<br><b>彩色粗点</b>：同一种颜色的两段海岸，原来是连在一起的一块大陆，后来裂开、漂远了（只标现在相隔 800 km 以上的）。' if os.environ.get("MARGINS") == "1" else '') + '</p>', page, flags=re.S)
open(sys.argv[1], "w").write(page)
print("page", len(page) // 1024, "KB")
