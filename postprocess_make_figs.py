#!/usr/bin/env python3
import re, os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

# ---------- filename patterns ----------
SUM_RX_BB = re.compile(r"disc_bb_Tb(?P<Tb>\d+)_summary\.txt$")
SUM_RX_DB = re.compile(r"disc_db_td(?P<td>\d+)_Tb(?P<Tb>\d+)_A(?P<A>[\d.]+)_summary\.txt$")
CSV_RX_BB = re.compile(r"disc_bb_Tb(?P<Tb>\d+)\.csv$")
CSV_RX_DB = re.compile(r"disc_db_td(?P<td>\d+)_Tb(?P<Tb>\d+)_A(?P<A>[\d.]+)\.csv$")

# ---------- parsing ----------
def parse_summary(path):
    """Support both old (E_act) and new (Wact/EPE/eta/...) formats."""
    txt = open(path, "r").read().strip()
    toks = dict(kv.split("=") for kv in re.findall(r"(\S+?=\S+)", txt))
    out = {}
    fn = os.path.basename(path)
    mBB, mDB = SUM_RX_BB.match(fn), SUM_RX_DB.match(fn)
    if mBB:
        out.update(sched="BB", Tb_ms=int(mBB.group("Tb")), td_ms=np.nan, A_scale=1.0, tag=fn.replace("_summary.txt",""))
    elif mDB:
        out.update(sched="DB", Tb_ms=int(mDB.group("Tb")), td_ms=int(mDB.group("td")),
                   A_scale=float(mDB.group("A")), tag=fn.replace("_summary.txt",""))
    else:
        return None

    def getf(key, fallback=None):
        if key in toks: 
            try: return float(toks[key])
            except: return fallback
        return fallback

    Wact = getf("Wact[J]", getf("E_act[J]"))
    EPE  = getf("EPE[J]")
    eta  = getf("eta")
    Fnpk = getf("Fn_peak[N]")
    Fnnm = getf("Fn_norm")
    dz   = getf("dz_apex[m]")
    dFdz = getf("dFdz_all[N/m]")
    dFdz_dw = getf("dFdz_dwell[N/m]")
    Imp  = getf("impulse[N*s]")
    G    = getf("G", (Fnpk/Imp if (Fnpk is not None and Imp and Imp>0) else None))
    HJ   = getf("HJ", (dz/Wact if (dz is not None and Wact and Wact>0) else None))
    sink = getf("sink[cm]")
    out.update(Wact_J=Wact, EPE_J=EPE, eta=eta, Fn_peak_N=Fnpk, Fn_norm=Fnnm,
               dz_apex_m=dz, dFdz_Npm=dFdz, dFdz_dwell_Npm=dFdz_dw,
               impulse_Ns=Imp, G=G, HJ=HJ, sink_cm=sink)
    return out

def load_summaries(folder):
    files = sorted(glob.glob(os.path.join(folder, "*_summary.txt")))
    rows = []
    for f in files:
        r = parse_summary(f)
        if r: rows.append(r)
    if not rows:
        print("[error] found no *_summary.txt")
        sys.exit(1)
    df = pd.DataFrame(rows)
    num_cols = ["Tb_ms","td_ms","A_scale","Wact_J","EPE_J","eta","Fn_peak_N","Fn_norm","dz_apex_m",
                "dFdz_Npm","dFdz_dwell_Npm","impulse_Ns","G","HJ","sink_cm"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def attach_baselines(df):
    bb = df[df.sched=="BB"].copy()
    if bb.empty:
        raise RuntimeError("No BB baselines found.")
    base = bb[["Tb_ms","Wact_J","eta","dz_apex_m","Fn_peak_N","Fn_norm","G","HJ"]]
    base = base.rename(columns={c:f"base_{c}" for c in base.columns if c!="Tb_ms"})
    dfm = df.merge(base, on="Tb_ms", how="left")
    def pct(a,b): 
        return np.where(np.isfinite(b) & (b!=0), 100.0*(a-b)/b, np.nan)
    dfm["dz_gain_pct"]   = pct(dfm["dz_apex_m"], dfm["base_dz_apex_m"])
    dfm["eta_gain_pct"]  = pct(dfm["eta"],        dfm["base_eta"])
    dfm["Fn_change_pct"] = pct(dfm["Fn_peak_N"],  dfm["base_Fn_peak_N"])
    dfm["E_rel_err_pct"] = pct(dfm["Wact_J"],     dfm["base_Wact_J"])
    dfm["G_change_pct"]  = pct(dfm["G"],          dfm["base_G"])
    dfm["HJ_change_pct"] = pct(dfm["HJ"],         dfm["base_HJ"])
    return dfm

# ---------- figures ----------
def make_heatmaps(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    dd = df[(df.sched=="DB")].copy()  # include td=0 here if present
    if dd.empty: return
    for col, fname, title in [
        ("dz_gain_pct", "heatmap_apex_delta.png", "Δ z_apex vs (Tb, td) [% vs BB]"),
        ("G",           "heatmap_pir.png", "Peak-to-Impulse Ratio (PIR) = Fn_peak / Impulse [s⁻¹] (lower is better)"),
     ]:
        piv = dd.pivot_table(index="td_ms", columns="Tb_ms", values=col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6,3.8))
        im = ax.imshow(piv.values, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(piv.shape[1]), labels=[str(c) for c in piv.columns])
        ax.set_yticks(np.arange(piv.shape[0]), labels=[str(r) for r in piv.index])
        ax.set_xlabel("Tb [ms]"); ax.set_ylabel("td [ms]"); ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i,j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:,.1f}", ha="center", va="center", fontsize=8, color="w")
        fig.savefig(os.path.join(outdir, fname)); plt.close(fig)

def load_step_csvs(folder):
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    idx = {}
    for f in csvs:
        bn = os.path.basename(f)
        if CSV_RX_BB.match(bn) or CSV_RX_DB.match(bn):
            idx[bn] = f
    return idx

def make_overlays(df, csv_idx, outdir):
    os.makedirs(outdir, exist_ok=True)
    for Tb, g in df.groupby("Tb_ms"):
        fig, ax = plt.subplots(figsize=(6,3.8))
        base = g[g.sched=="BB"]
        if not base.empty:
            bn = f"disc_bb_Tb{int(Tb)}.csv"
            if bn in csv_idx:
                c = pd.read_csv(csv_idx[bn])
                if {"t","Wact_cum_J"}.issubset(c.columns):
                    ax.plot(c["t"], c["Wact_cum_J"], linewidth=2.0, label=f"BB Tb{int(Tb)}")
        for _,r in g[g.sched=="DB"].iterrows():
            bn = f"disc_db_td{int(r.td_ms)}_Tb{int(Tb)}_A{r.A_scale:.2f}.csv"
            if bn in csv_idx:
                c = pd.read_csv(csv_idx[bn])
                if {"t","Wact_cum_J"}.issubset(c.columns):
                    ax.plot(c["t"], c["Wact_cum_J"], alpha=0.9, label=f"DB td{int(r.td_ms)} A{r.A_scale:.2f}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("Cumulative work W_act [J]")
        ax.set_title(f"Equal-work overlay (Tb={int(Tb)} ms)")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        fig.savefig(os.path.join(outdir, f"overlay_Tb{int(Tb)}.png")); plt.close(fig)

def make_envelope_faceted(df, outdir):
    """
    Faceted envelope: one panel per Tb. 
    - BB baseline as black circle.
    - DB points colored by td (nonzero only).
    - No lines, no per-point labels. Shared x/y limits.
    """
    os.makedirs(outdir, exist_ok=True)
    # Panels in increasing Tb order, only those that have a BB baseline
    tbs = sorted(df[df.sched=="BB"]["Tb_ms"].unique().tolist())
    if not tbs:
        return

    # Global axis limits for readability
    x_all = df["Fn_norm"].dropna().values
    y_all = df["dz_apex_m"].dropna().values
    if x_all.size==0 or y_all.size==0: 
        return
    xpad = 0.05*(np.nanmax(x_all)-np.nanmin(x_all) if np.nanmax(x_all)>np.nanmin(x_all) else 1)
    ypad = 0.05*(np.nanmax(y_all)-np.nanmin(y_all) if np.nanmax(y_all)>np.nanmin(y_all) else 1)
    xlim = (np.nanmin(x_all)-xpad, np.nanmax(x_all)+xpad)
    ylim = (0.0, np.nanmax(y_all)+ypad)

    # Color map by all nonzero dwells across the dataset (consistent mapping)
    all_nonzero_td = sorted([int(x) for x in df.loc[(df.sched=="DB") & (df["td_ms"]>0), "td_ms"].dropna().unique()])
    cmap = plt.cm.get_cmap("tab10", max(1,len(all_nonzero_td)))
    td2color = {td: cmap(ii % 10) for ii,td in enumerate(all_nonzero_td)}

    n = len(tbs)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.8*nrows), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(nrows, ncols)

    # Draw panels
    for k, Tb in enumerate(tbs):
        i, j = divmod(k, ncols)
        ax = axes[i, j]
        g = df[df["Tb_ms"]==Tb].copy()

        # BB baseline(s)
        bb = g[g.sched=="BB"]
        if not bb.empty:
            ax.scatter(bb["Fn_norm"], bb["dz_apex_m"], s=45, marker="o", color="black", label="_nolegend_")

        # DB (td>0 only)
        dd = g[(g.sched=="DB") & (g["td_ms"]>0)].copy()
        for td in sorted(dd["td_ms"].unique()):
            sub = dd[dd["td_ms"]==td]
            ax.scatter(sub["Fn_norm"], sub["dz_apex_m"], s=35, marker="o", color=td2color.get(int(td), "C0"),
                       label=f"td={int(td)} ms")

        ax.set_title(f"Tb={int(Tb)} ms")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        if i==nrows-1:
            ax.set_xlabel(r"Normalized peak $F_n/(mg)$")
            # small bump to avoid legend/tick overlap in tight layouts
            ax.xaxis.set_label_coords(0.5, -0.12)
        if j==0:
            ax.set_ylabel("Apex height [m]")

    # Hide any empty axes (if grid not filled)
    for k in range(n, nrows*ncols):
        i, j = divmod(k, ncols)
        axes[i, j].set_visible(False)

    # Single shared legend (unique dwells only)
    # Build handles manually to keep order & avoid duplicates
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], linestyle="None", marker="o", color=td2color[td], label=f"td={td} ms") for td in all_nonzero_td]
    # Place legend below the subplots, clear of titles
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Design envelope: height vs normalized peak load", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.12)

    fig.savefig(os.path.join(outdir, "envelope_apex_vs_FnNorm_faceted_refined.png"))
    fig.savefig(os.path.join(outdir, "envelope_apex_vs_FnNorm_faceted_refined.pdf"))
    plt.close(fig)

def write_rulebox(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    lines = []
    lines.append("Recommended operating rules (auto-derived):")
    lines.append("— Short bursts with modest dwell minimize peak loads for small height penalty.")
    lines.append("— Prefer lower PIR (Fn_peak / Impulse) for comparable height.")
    best = []
    for Tb in sorted(df["Tb_ms"].unique()):
        g = df[df["Tb_ms"]==Tb]
        bb = g[g.sched=="BB"]
        if bb.empty: 
            continue
        # choose DB with ≥20% peak reduction (Fn_change_pct ≤ −20) and height loss ≤5%
        cand = g[(g.sched=="DB") & (g["td_ms"]>0) & (g["Fn_change_pct"]<=-20.0) & (g["dz_gain_pct"]>=-5.0)]
        if not cand.empty:
            # prefer greatest peak reduction, then least height loss
            r = cand.sort_values(["Fn_change_pct","dz_gain_pct"], ascending=[True, False]).iloc[0]
            best.append(r)
            lines.append(f"  Tb={int(Tb)} ms: td={int(r.td_ms)} ms → Δapex={r.dz_gain_pct:+.1f}%, ΔFn={r.Fn_change_pct:+.1f}% (A*={r.A_scale:.2f}).")
    if not best:
        lines.append("  (No DB setting met the default constraints; relax thresholds or inspect envelope.)")
    with open(os.path.join(outdir, "rulebox.txt"), "w") as f:
        f.write("\n".join(lines))

# ---------- main ----------
def main():
    if len(sys.argv)<2:
        print("Usage: python3 postprocess_make_figs_refined.py <folder> [--out figures_refined]")
        sys.exit(1)
    folder = sys.argv[1]
    outroot = "figures_refined"
    if "--out" in sys.argv:
        outroot = sys.argv[sys.argv.index("--out")+1]

    df = load_summaries(folder)
    df = attach_baselines(df)
    df.sort_values(["Tb_ms","sched","td_ms","A_scale"], inplace=True)
    out_csv = os.path.join(folder, "design_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"[info] Wrote {out_csv} ({len(df)} rows)")

    # Equal-work report
    def flag(err): return ("OK" if (abs(err)<=1.0) else "WARN")
    eq = df[["tag","sched","Tb_ms","td_ms","A_scale","Wact_J","base_Wact_J","E_rel_err_pct"]].copy()
    print("\nEqual-work check (|error| ≤ 1% target):")
    for _,r in eq.sort_values(["Tb_ms","sched","td_ms","A_scale"]).iterrows():
        if r.sched=="BB":
            print(f"  BB  Tb={int(r.Tb_ms):>3}  A*=1.00  W={r.Wact_J:.4f} J   [{flag(0)}]")
        else:
            print(f"  DB  Tb={int(r.Tb_ms):>3} td={int(r.td_ms):>2} A*={r.A_scale:.2f}  "
                  f"W={r.Wact_J:.4f} J vs {r.base_Wact_J:.4f} J  err={r.E_rel_err_pct:+.2f}%  [{flag(r.E_rel_err_pct)}]")

    figs_dir = outroot
    os.makedirs(figs_dir, exist_ok=True)

    make_heatmaps(df, figs_dir)
    make_envelope_faceted(df, figs_dir)

    csv_idx = load_step_csvs(folder)
    make_overlays(df, csv_idx, os.path.join(figs_dir, "overlays"))
    write_rulebox(df, figs_dir)

    print(f"\n[done] Figures in: {figs_dir}")
    print("      Heatmaps:   heatmap_apex_delta.png, heatmap_pir.png")
    print("      Envelope:   envelope_apex_vs_FnNorm_faceted_refined.(png|pdf)")
    print("      Overlays:   overlays/overlay_TbXX.png")
    print("      Rule box:   rulebox.txt")

if __name__ == "__main__":
    main()
