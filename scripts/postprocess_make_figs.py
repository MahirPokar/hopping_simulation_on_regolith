#!/usr/bin/env python3
import re, os, sys, glob, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

SUM_RX_BB = re.compile(r"disc_bb_Tb(?P<Tb>\d+)_summary\.txt$")
SUM_RX_DB = re.compile(r"disc_db_td(?P<td>\d+)_Tb(?P<Tb>\d+)_A(?P<A>[\d.]+)_summary\.txt$")
CSV_RX_BB = re.compile(r"disc_bb_Tb(?P<Tb>\d+)\.csv$")
CSV_RX_DB = re.compile(r"disc_db_td(?P<td>\d+)_Tb(?P<Tb>\d+)_A(?P<A>[\d.]+)\.csv$")

def parse_summary(path):
    """Support BOTH old (E_act) and new (Wact/EPE/eta/...) formats."""
    txt = open(path, "r").read().strip()
    # normalize whitespace to "key=value" tokens
    toks = dict(kv.split("=") for kv in re.findall(r"(\S+?=\S+)", txt))
    out = {}
    # file-derived
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
        if key in toks: return float(toks[key])
        return fallback

    # new keys
    Wact = getf("Wact[J]", getf("E_act[J]"))
    EPE  = getf("EPE[J]")
    eta  = getf("eta")
    Fnpk = getf("Fn_peak[N]")
    Fnnm = getf("Fn_norm")
    dz   = getf("dz_apex[m]", getf("dz_apex[m]"))  # both formats used same key
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
    # enforce types
    num_cols = ["Tb_ms","td_ms","A_scale","Wact_J","EPE_J","eta","Fn_peak_N","Fn_norm","dz_apex_m",
                "dFdz_Npm","dFdz_dwell_Npm","impulse_Ns","G","HJ","sink_cm"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def attach_baselines(df):
    bb = df[df.sched=="BB"].copy()
    if bb.empty:
        raise RuntimeError("No BB baselines found.")
    # ensure one baseline per Tb
    base = bb[["Tb_ms","Wact_J","eta","dz_apex_m","Fn_peak_N","Fn_norm","G","HJ"]]
    base = base.rename(columns={c:f"base_{c}" for c in base.columns if c!="Tb_ms"})
    dfm = df.merge(base, on="Tb_ms", how="left")
    # deltas vs baseline
    def pct(a,b): 
        return np.where(np.isfinite(b) & (b!=0), 100.0*(a-b)/b, np.nan)
    dfm["dz_gain_pct"]   = pct(dfm["dz_apex_m"], dfm["base_dz_apex_m"])
    dfm["eta_gain_pct"]  = pct(dfm["eta"],        dfm["base_eta"])
    dfm["Fn_change_pct"] = pct(dfm["Fn_peak_N"],  dfm["base_Fn_peak_N"])
    dfm["E_rel_err_pct"] = pct(dfm["Wact_J"],     dfm["base_Wact_J"])
    dfm["G_change_pct"]  = pct(dfm["G"],          dfm["base_G"])
    dfm["HJ_change_pct"] = pct(dfm["HJ"],         dfm["base_HJ"])
    return dfm

def make_heatmaps(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    dd = df[df.sched=="DB"].copy()
    if dd.empty: return
    for col, fname, title in [
        ("dz_gain_pct", "heatmap_apex_delta.png", "Δ z_apex vs (Tb, td) [% vs BB]"),
        ("G",           "heatmap_gentleness.png", "Gentleness G = Fn_peak / Impulse (lower is better)"),
    ]:
        piv = dd.pivot_table(index="td_ms", columns="Tb_ms", values=col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6,3.8))
        im = ax.imshow(piv.values, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(piv.shape[1]), labels=[str(c) for c in piv.columns])
        ax.set_yticks(np.arange(piv.shape[0]), labels=[str(r) for r in piv.index])
        ax.set_xlabel("Tb [ms]"); ax.set_ylabel("td [ms]"); ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.grid(False)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i,j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:,.1f}", ha="center", va="center", fontsize=8, color="w")
        fig.savefig(os.path.join(outdir, fname)); plt.close(fig)

import math, os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def make_envelope(df, outdir, fname="envelope_apex_vs_FnNorm_faceted_pro.png"):
    os.makedirs(outdir, exist_ok=True)

    # Panels to make
    Tbs = sorted(df["Tb_ms"].dropna().unique().astype(int))
    ncols = 2
    nrows = math.ceil(len(Tbs) / ncols)

    # Global limits (shared axes)
    x = df["Fn_norm"].dropna().values
    y = df["dz_apex_m"].dropna().values
    if len(x) == 0 or len(y) == 0:
        raise RuntimeError("Fn_norm or dz_apex_m not found/empty in dataframe.")
    xpad = 0.05 * (x.max() - x.min() if x.max() > x.min() else 1.0)
    ypad = 0.05 * (y.max() - y.min() if y.max() > y.min() else 1.0)
    xmin, xmax = x.min() - xpad, x.max() + xpad
    ymin, ymax = y.min() - ypad, y.max() + ypad

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.2, 3.6 * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    # Style maps (no td=0)
    color_by_td = {20: "#1f77b4", 40: "#ff7f0e"}  # green / red
    marker_bb, size_bb = "o", 70                  # baseline: black circle
    marker_db, size_db = "o", 55                  # DB points: colored circles

    for ax, Tb in zip(axes, Tbs):
        sub = df[df.Tb_ms == Tb].copy()

        # Baseline (one point)
        bb = sub[sub.sched == "BB"].iloc[0]
        ax.scatter(bb.Fn_norm, bb.dz_apex_m, s=size_bb, marker=marker_bb, c="black",
                   label="BB baseline", zorder=3)

        # DB points for td ∈ {20, 40}
        db_sub = sub[(sub.sched == "DB") & (sub["td_ms"].isin([20, 40]))]
        for td, grp in db_sub.groupby("td_ms"):
            color = color_by_td[int(td)]
            ax.scatter(grp.Fn_norm, grp.dz_apex_m, s=size_db, marker=marker_db,
                       color=color, edgecolor="none", zorder=3)

        ax.set_title(f"Tb={int(Tb)} ms")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Remove any empty axes
    for ax in axes[len(Tbs):]:
        fig.delaxes(ax)

    # Global labels
    fig.supxlabel(r"Normalized peak $F_n/(mg)$", y=0.10)
    fig.supylabel("Apex height [m]")

    # Legend at bottom (no overlap with titles)
    handles = [
        mlines.Line2D([], [], color="black", marker=marker_bb, linestyle="None",
                      markersize=7, label="BB baseline")
    ]
    if (df["td_ms"] == 20).any():
        handles.append(
            mlines.Line2D([], [], color=color_by_td[20], marker=marker_db, linestyle="None",
                          markersize=6, label="DB, dwell 20 ms")
        )
    if (df["td_ms"] == 40).any():
        handles.append(
            mlines.Line2D([], [], color=color_by_td[40], marker=marker_db, linestyle="None",
                          markersize=6, label="DB, dwell 40 ms")
        )

    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.0))

    # Make room for bottom legend and panel titles
    fig.subplots_adjust(bottom=0.16, top=0.94, wspace=0.18, hspace=0.28)

    # Save PNG + vector PDF
    out_png = os.path.join(outdir, fname)
    out_pdf = os.path.join(outdir, os.path.splitext(fname)[0] + ".pdf")
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)  

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
        # baseline
        base = g[g.sched=="BB"]
        if not base.empty:
            bn = f"disc_bb_Tb{int(Tb)}.csv"
            if bn in csv_idx:
                c = pd.read_csv(csv_idx[bn])
                if {"t","Wact_cum_J"}.issubset(c.columns):
                    ax.plot(c["t"], c["Wact_cum_J"], linewidth=2.0, label=f"BB Tb{int(Tb)}")
        # DBs
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

def write_rulebox(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    lines = []
    lines.append("Recommended operating rules (auto-derived):")
    lines.append("— Short bursts with modest dwell minimize peak loads for small height penalty.")
    best = []
    for Tb, g in df.groupby("Tb_ms"):
        bb = g[g.sched=="BB"].iloc[0]
        # choose DB with ≥20% peak reduction (Fn_change_pct ≤ −20) and height loss ≤5% (dz_gain ≥ −5)
        cand = g[(g.sched=="DB") & (g["Fn_change_pct"]<=-20.0) & (g["dz_gain_pct"]>=-5.0)]
        if not cand.empty:
            r = cand.sort_values(["Fn_change_pct","-dz_gain_pct".replace("-","")], ascending=[True, False]).iloc[0]
            best.append(r)
            lines.append(f"  Tb={int(Tb)} ms: td={int(r.td_ms)} ms → Δapex={r.dz_gain_pct:+.1f}%,"
                         f" ΔFn={r.Fn_change_pct:+.1f}% (A*={r.A_scale:.2f}).")
    if not best:
        lines.append("  (No DB setting met the default constraints; relax thresholds or inspect envelope.)")
    with open(os.path.join(outdir, "rulebox.txt"), "w") as f:
        f.write("\n".join(lines))

def main():
    if len(sys.argv)<2:
        print("Usage: python3 postprocess_make_figs.py <folder_with_summaries_and_csvs> [--out figures]")
        sys.exit(1)
    folder = sys.argv[1]
    outroot = "figures"
    if "--out" in sys.argv:
        outroot = sys.argv[sys.argv.index("--out")+1]

    df = load_summaries(folder)
    df = attach_baselines(df)
    # save clean table
    df.sort_values(["Tb_ms","sched","td_ms","A_scale"], inplace=True)
    df.to_csv(os.path.join(folder, "design_sweep.csv"), index=False)
    print(f"[info] Wrote {os.path.join(folder,'design_sweep.csv')} ({len(df)} rows)")

    # equal-work report
    def flag(err):
        return ("OK" if (abs(err)<=1.0) else "WARN")
    eq = df[["tag","sched","Tb_ms","td_ms","A_scale","Wact_J","base_Wact_J","E_rel_err_pct"]].copy()
    print("\nEqual-work check (|error| ≤ 1% target):")
    for _,r in eq.sort_values(["Tb_ms","sched","td_ms","A_scale"]).iterrows():
        if r.sched=="BB":
            print(f"  BB  Tb={int(r.Tb_ms):>3}  A*=1.00  W={r.Wact_J:.4f} J   [{flag(0)}]")
        else:
            print(f"  DB  Tb={int(r.Tb_ms):>3} td={int(r.td_ms):>2} A*={r.A_scale:.2f}  "
                  f"W={r.Wact_J:.4f} J vs {r.base_Wact_J:.4f} J  err={r.E_rel_err_pct:+.2f}%  [{flag(r.E_rel_err_pct)}]")

    # figures
    figs_dir = outroot
    make_heatmaps(df, figs_dir)
    make_envelope(df, figs_dir)
    csv_idx = load_step_csvs(folder)
    make_overlays(df, csv_idx, os.path.join(figs_dir, "overlays"))
    write_rulebox(df, figs_dir)

    print(f"\n[done] Figures in: {figs_dir}")
    print("      Heatmaps:   heatmap_apex_delta.png, heatmap_gentleness.png")
    print("      Envelope:   envelope_apex_vs_FnNorm.png")
    print("      Overlays:   overlays/overlay_TbXX.png")
    print("      Rule box:   rulebox.txt")

if __name__ == "__main__":
    main()
