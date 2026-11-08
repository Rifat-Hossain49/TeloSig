"""Stage 4: paper figures (PDF + PNG) and LaTeX tables from results/.

Palette: reference categorical slots 1-3 (validated all-pairs in light mode; see the dataviz reference
palette) + muted gray for references and published methods. Print figures, so series are labelled directly
and every figure has a legend or direct labels.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID, AXIS = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,                 # embed TrueType, not Type 3, fonts in vector output
})
SIG = f"LR (signature, K={C.SIG_K})"
TW = "LR (whole transcriptome)"
TITLE = {"alt": "ALT, gliomas + sarcomas", "alt_noATRX": "ALT without ATRX/DAXX", "alt_pan": "ALT, pan-cancer",
         "alt_pheno": "ALT by WGS phenotype", "tel": "TERT expression proxy (TERT excluded)",
         "tl": "Telomere length ratio"}
# Display names for figures and tables. Result files keep the original model keys; "GBM (panel)" is
# histogram gradient boosting, shown as HGB so it cannot be confused with glioblastoma (GBM).
DISPLAY = {"GBM (panel)": "HGB (panel)", "Locked telomerase signature": "Locked TERT-proxy signature",
           "Locked alt signature": "Locked ALT signature (gliomas + sarcomas)",
           "Locked alt_pan signature": "Locked ALT signature (pan-cancer)"}


def disp(model):
    return DISPLAY.get(model, model)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIGURES / f"{name}.{ext}")
    plt.close(fig)


def summary(task):
    p = C.RESULTS / f"{task}_summary.csv"
    return pd.read_csv(p).set_index("model") if p.exists() else None


def csv(name):
    p = C.RESULTS / name
    if not p.exists() or p.stat().st_size < 5:      # e.g. no enriched terms -> empty file
        return None
    return pd.read_csv(p)


def dot_ci(ax, rows, key="within_AUROC", chance=0.5):
    """rows: list of (label, series-with key/key_lo/key_hi, color). Top row first."""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r, c) in zip(y, rows):
        ax.plot([r[f"{key}_lo"], r[f"{key}_hi"]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r[key], yi, "o", ms=6, color=c, mec="white", mew=1.5)
        ax.annotate(f"{r[key]:.3f}", (r[f"{key}_hi"], yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(chance, color=AXIS, lw=1)
    lo = min(r[1][f"{key}_lo"] for r in rows)
    hi = max(r[1][f"{key}_hi"] for r in rows)
    ax.set_xlim(min(chance, lo) - 0.05, hi + 0.07)      # keep the chance line clear of the axis spine
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------------
def fig_study_design():
    """Study design: internal benchmark, models fixed before validation, validation tiers."""
    from matplotlib.patches import Ellipse
    W, H = 7.2, 4.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    LIGHT, DARK, PALE = "#eaf2fb", "#1f3b63", "#f1f0ec"

    def box(x, y, w, h, title, body, edge=BLUE, face=LIGHT, title_color=INK, body_color=INK2, num=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    facecolor=face, edgecolor=edge, linewidth=1.1))
        top = (x + w / 2, y + h)                         # text offsets in points keep spacing font-relative
        ax.annotate(title, top, xytext=(0, -7), textcoords="offset points", ha="center", va="top",
                    fontsize=8.2, weight="bold", color=title_color)
        ax.annotate(body, top, xytext=(0, -21), textcoords="offset points", ha="center", va="top",
                    fontsize=6.6, color=body_color, linespacing=1.35)
        if num is not None:
            r = 0.016
            ax.add_patch(Ellipse((x + 0.006, y + h - 0.004), 2 * r, 2 * r * W / H, color=DARK, zorder=5))
            ax.text(x + 0.006, y + h - 0.004, str(num), ha="center", va="center", fontsize=6.3,
                    color="white", weight="bold", zorder=6)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     color=DARK, linewidth=1.0, shrinkA=0, shrinkB=0))

    ax.text(0.015, 0.985, "Internal benchmark", fontsize=7.2, color=INK2, weight="bold", va="top")
    top_y, top_h, w, step = 0.71, 0.21, 0.215, 0.245
    stages = [("TCGA data", "RNA-seq + ALT /\nTERT proxy labels"),
              ("Preprocessing", "TERT excluded\nfrom predictors\nParameters learned\nfrom training folds"),
              ("Nested benchmark", "Repeated nested\ncross-validation\nEvaluate within\neach cancer type"),
              ("Confounding checks", "Molecular subtype\nCopy number +\nchromosome regions")]
    xs = [0.02 + i * step for i in range(4)]
    for i, (x, (title, body)) in enumerate(zip(xs, stages)):
        box(x, top_y, w, top_h, title, body, num=i + 1)
    for x in xs[:-1]:
        arrow(x + w + 0.003, top_y + top_h / 2, x + step - 0.003, top_y + top_h / 2)

    fx, fw, fy, fh = 0.28, 0.44, 0.43, 0.15
    ax.plot([xs[3] + w / 2, xs[3] + w / 2, 0.5], [top_y, top_y - 0.04, top_y - 0.04], color=DARK, lw=1.0)
    arrow(0.5, top_y - 0.04, 0.5, fy + fh)
    box(fx, fy, fw, fh, "Models fixed before validation",
        "Scaling, coefficients and decision threshold\nare taken from training and kept unchanged",
        edge=DARK, face=DARK, title_color="white", body_color="white")

    vy, vh, vw = 0.175, 0.175, 0.29
    vx = [0.03, 0.355, 0.68]
    branch_y = fy - 0.04
    ax.plot([0.5, 0.5], [fy, branch_y], color=DARK, lw=1.0)
    ax.plot([vx[0] + vw / 2, vx[2] + vw / 2], [branch_y, branch_y], color=DARK, lw=1.0)
    for x in vx:
        arrow(x + vw / 2, branch_y, x + vw / 2, vy + vh)
    validation = [("Genomic validation", "ALT evidence from WGS\nTERT DNA alterations", ORANGE),
                  ("Direct assays", "qTRAP telomerase activity\nC-circle assay", AQUA),
                  ("RNA quantification check", "Same tumours, TCGA vs PCAWG\nTests technical reproducibility", BLUE)]
    for x, (title, body, edge) in zip(vx, validation):
        box(x, vy, vw, vh, title, body, edge=edge, face="white")

    sx, sw, sy, sh = 0.355, 0.29, 0.005, 0.135
    arrow(vx[1] + vw / 2, vy, vx[1] + vw / 2, sy + sh)
    box(sx, sy, sw, sh, "Secondary analyses", "Clinical outcomes\nTelomere length from sequencing",
        edge=AXIS, face=PALE)
    save(fig, "fig1_study_design")


def fig_feature_sets():
    """Which gene set carries the signal? References and published methods vs curated panel, compact
    signature and the whole transcriptome (within-cancer AUROC, 95% CI)."""
    tasks = [t for t in ("alt", "tel") if summary(t) is not None]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 3.2), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        s = summary(t)
        rows = []
        for m, lab in [("Cancer + glioma subtype only", "Cancer type + glioma subtype"),
                       ("Barthel 2017 score (published)", "Barthel 2017 score (43 genes)"),
                       ("LR (EXTEND gene set, TERT removed)", "EXTEND gene set, TERT removed")]:
            if m in s.index:
                rows.append((lab, s.loc[m], GRAY))
        panel = [m for m in s.index if m.endswith("(panel)")]
        rows.append(("Telomere panel, LR (63 genes)", s.loc["LR (panel)"], BLUE))
        best = s.loc[panel, "within_AUROC"].idxmax()
        if best != "LR (panel)":
            rows.append((f"Telomere panel, best model: {best.split(' (')[0]}", s.loc[best], BLUE))
        if SIG in s.index:
            rows.append((f"Compact signature ({C.SIG_K} genes)", s.loc[SIG], ORANGE))
        if TW in s.index:
            rows.append(("Whole transcriptome (all protein-coding)", s.loc[TW], BLUE))
        dot_ci(ax, rows)
        ax.set_xlabel("Within-cancer AUROC (95% CI)")
        ax.set_title(TITLE[t], fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig2_feature_sets")


def fig_signature_curve():
    tasks = [t for t in ("alt", "tel") if (C.RESULTS / f"{t}_signature_curve_ci.csv").exists()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 2.8), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        cu = csv(f"{t}_signature_curve_ci.csv")
        s = summary(t)
        for mth, color, lab in [("l1", ORANGE, "L1 selection"), ("univariate", BLUE, "Univariate selection")]:
            c = cu[cu.method == mth].sort_values("k")
            ax.fill_between(c.k, c.within_AUROC_lo, c.within_AUROC_hi, color=color, alpha=0.15, lw=0)
            ax.plot(c.k, c.within_AUROC, "o-", color=color, lw=2, ms=5, mec="white", mew=1, label=lab)
        for m, lab in [("LR (panel)", "Telomere panel (63)"), (TW, "Whole transcriptome")]:
            if s is not None and m in s.index:
                v = s.loc[m, "within_AUROC"]
                ax.axhline(v, color=INK2 if m == TW else GRAY, lw=1)
                ax.annotate(f"{lab} {v:.3f}", (cu.k.min(), v), xytext=(0, 3), textcoords="offset points",
                            ha="left", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(sorted(cu.k.unique()), [str(k) for k in sorted(cu.k.unique())])
        ax.minorticks_off()
        ax.set_xlabel("Genes in signature (selected inside each training fold)")
        ax.set_ylabel("Within-cancer AUROC")
        ax.set_title(TITLE[t], fontsize=8.5, color=INK, loc="left")
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig3_signature_size")


def fig_subtype_control():
    ws = csv("alt_within_subtype.csv")
    if ws is None or ws.empty:
        return
    models = [("Cancer + glioma subtype only", GRAY), ("LR (panel)", BLUE), (SIG, ORANGE), (TW, AQUA)]
    models = [(m, c) for m, c in models if m in set(ws.model)]
    subs = ws.drop_duplicates("subtype")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    width = 0.8 / len(models)
    for i, (m, color) in enumerate(models):
        v = ws[ws.model == m].set_index("subtype").reindex(subs.subtype)["AUROC"].values
        x = np.arange(len(subs)) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, v - 0.5, width * 0.92, bottom=0.5, color=color, label=m)
    ax.axhline(0.5, color=AXIS, lw=1)
    ax.set_xticks(np.arange(len(subs)),
                  [f"{s.replace('GBM_LGG.', '')}\n(n={n}, ALT={p})" for s, n, p in zip(subs.subtype, subs.n, subs.n_pos)])
    ax.set_ylabel("AUROC within subtype")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("ALT detection inside single molecular subtypes", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_subtype_control")


def fig_validation():
    """External labels (held-out PCAWG donors) and agreement across two RNA-seq quantification pipelines."""
    ext, plat = csv("validation_external.csv"), csv("validation_platform_scores.csv")
    panels = (1 if ext is not None else 0) + (2 if plat is not None else 0)
    if not panels:
        return
    fig, axes = plt.subplots(1, panels, figsize=(3.5 * panels, 3.0), squeeze=False)
    axes = list(axes[0])
    if ext is not None:
        ax = axes.pop(0)
        e = ext[ext.task == "alt"]
        if "scope" in e:
            e = e[e.scope.fillna("all") == "all"]
        colors = {"Cancer type only": GRAY, "Barthel genotype label": GRAY, "LR (panel)": BLUE, "GBM (panel)": BLUE}
        rows = [(disp(r.model), r.rename({"AUROC": "k", "AUROC_lo": "k_lo", "AUROC_hi": "k_hi"}),
                 ORANGE if r.model.startswith("Signature") else colors.get(r.model, BLUE)) for _, r in e.iterrows()]
        dot_ci(ax, rows, key="k")
        ax.set_xlabel("AUROC vs WGS telomere-sequence ALT (95% CI)")
        n, npos = int(e.n.iloc[0]), int(e.n_pos.iloc[0])
        ax.set_title(f"Held-out PCAWG donors, gliomas + sarcomas\nn={n}, ALT={npos}", fontsize=8, color=INK, loc="left")
    if plat is not None:
        for t, ax in zip(["alt", "tel"], axes):
            p = plat[plat.task == t]
            if p.empty:
                continue
            rho = p.score_toil.corr(p.score_pcawg, method="spearman")
            for lab, color, name in [(0, GRAY, "label negative"), (1, ORANGE if t == "alt" else BLUE, "label positive")]:
                k = p.label == lab
                ax.scatter(p.score_toil[k], p.score_pcawg[k], s=9, color=color, alpha=0.7, lw=0, label=name)
            ax.set_xlabel("Signature score, TCGA pipeline (RSEM TPM)")
            ax.set_ylabel("Signature score, PCAWG pipeline (FPKM-UQ)")
            ax.set_title(f"{TITLE[t]}\nsame tumors, n={len(p)}, Spearman ρ={rho:.2f}", fontsize=8, color=INK, loc="left")
            ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_validation")


def fig_cell_lines():
    legacy = csv("validation_celllines.csv")
    wu = csv("validation_wu2025_celllines.csv")
    groups = []
    if wu is not None and not wu.empty:
        q = wu[(wu.task == "tel") & (wu.metric == "Spearman") & (wu.scope == "all")].copy()
        if not q.empty:
            q = q.rename(columns={"estimate": "spearman"})
            groups.append(("Wu 2025 qTRAP", q[["model", "n", "spearman", "lo", "hi"]], BLUE))
    if legacy is not None and not legacy.empty:
        for panel, q in legacy.groupby("panel"):
            groups.append((str(panel), q[["model", "n", "spearman", "lo", "hi"]], ORANGE))
    if not groups:
        return
    heights = [max(1.3, 0.34 * len(q) + 0.7) for _, q, _ in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(5.2, sum(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (label, sub, color) in zip(axes[:, 0], groups):
        sub = sub.sort_values("spearman").reset_index(drop=True)
        pos = np.arange(len(sub))
        ax.errorbar(sub.spearman, pos, xerr=[sub.spearman - sub.lo, sub.hi - sub.spearman],
                    fmt="o", ms=4.5, color=color, ecolor=INK2, elinewidth=0.8, capsize=1.8)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(pos, [disp(m) for m in sub.model])
        ax.set_xlim(-1, 1)
        ax.set_title(f"{label} (up to n={int(sub['n'].max())})", fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("Spearman rho (95% CI)")
    fig.suptitle("Independent enzymatic telomerase-activity validation", fontsize=8.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig6_cell_lines")


def fig_survival():
    """Forest plot of the primary survival models; estimates are listed in a separate aligned column."""
    s = csv("validation_survival.csv")
    if s is None or s.empty:
        return
    s = s[s.adjustment != "unadjusted"].copy()
    if "analysis_role" in s:
        s = s[s.analysis_role == "primary"]
    s["label"] = s.task.map({"alt": "ALT score", "tel": "TERT proxy score"}) + " | " + s.group + " | " + \
        s.endpoint + " | " + s.adjustment
    ys, y, prev = [], 0.0, None
    for _, r in s.iterrows():
        key = (r.task, r.group)
        if prev is not None and key != prev:
            y += 0.7                                     # visual gap between cancer-type groups
        ys.append(y)
        y += 1.0
        prev = key
    ys = -np.array(ys)
    fig, ax = plt.subplots(figsize=(7.0, 0.3 * len(s) + 1.3))
    fig.subplots_adjust(left=0.36, right=0.70)
    for yi, (_, r) in zip(ys, s.iterrows()):
        q = r.get("p_fdr", r.p)
        c = ORANGE if q < 0.05 else BLUE
        ax.plot([r.HR_lo, r.HR_hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot(r.HR_per_SD, yi, "s", ms=5, color=c, mec="white", mew=0.8)
        ax.text(1.04, yi, f"HR {r.HR_per_SD:.2f}   FDR {q:.3f}   {int(r.events):,} events",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.5, color=INK2)
    ax.text(1.04, ys.max() + 0.9, "Estimate", transform=ax.get_yaxis_transform(), va="center", fontsize=7,
            color=INK, weight="bold")
    ax.axvline(1, color=AXIS, lw=1)
    ax.set_xscale("log")
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3) if s.HR_lo.min() * 0.9 <= t <= s.HR_hi.max() * 1.1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(ys, s.label, fontsize=6.5)
    ax.set_ylim(ys.min() - 0.8, ys.max() + 0.8)
    ax.set_xlabel("Hazard ratio per SD of cross-validated score (95% CI)")
    ax.set_title("Association with clinical outcome within cancer types", fontsize=8.5, color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_survival")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------
def figS_all_models():
    tasks = [t for t in TITLE if summary(t) is not None]
    if not tasks:
        return
    fig, axes = plt.subplots(2, (len(tasks) + 1) // 2, figsize=(4.2 * ((len(tasks) + 1) // 2), 6.2), squeeze=False)
    for ax, t in zip(axes.ravel(), tasks):
        s = summary(t).drop(index="Cancer type only", errors="ignore")
        clf = "within_AUROC" in s.columns
        key = "within_AUROC" if clf else "within_Spearman"
        s = s.sort_values(key, ascending=False)
        rows = [(disp(m), r, ORANGE if m == SIG else (GRAY if ("only" in m or "published" in m) else BLUE))
                for m, r in s.iterrows()]
        dot_ci(ax, rows, key=key, chance=0.5 if clf else 0.0)
        ax.set_xlabel("Within-cancer AUROC" if clf else "Within-cancer Spearman ρ")
        ax.set_title(TITLE[t], fontsize=8, color=INK, loc="left")
    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "figS1_all_models")


def figS_per_cancer():
    s = summary("tel")
    if s is None:
        return
    cols = [c for c in s.columns if c.startswith("AUROC[")]
    models = [(m, c) for m, c in [("LR (panel)", BLUE), (SIG, ORANGE), (TW, AQUA)] if m in s.index]
    order = np.argsort(s.loc[models[-1][0], cols].values)
    fig, ax = plt.subplots(figsize=(3.8, 0.22 * len(cols) + 1.0))
    for (m, color), dy in zip(models, np.linspace(-0.2, 0.2, len(models))):
        ax.plot(s.loc[m, cols].values[order], np.arange(len(cols)) + dy, "o", ms=4.5, color=color, mec="white", mew=0.8, label=m)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.set_yticks(np.arange(len(cols)), np.array([c[6:-1] for c in cols])[order])
    ax.set_xlabel("Within-cancer AUROC")
    ax.set_title(TITLE["tel"] + ", per cancer type", fontsize=8, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    save(fig, "figS2_per_cancer_tel")


def figS_random_null():
    tasks = [t for t in ("alt", "tel") if (C.RESULTS / f"{t}_random_null.csv").exists()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.3 * len(tasks), 2.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        r = csv(f"{t}_random_null.csv")
        panel, rand = r.within_AUROC.iloc[0], r.within_AUROC.iloc[1:]
        p = (1 + (rand >= panel).sum()) / (1 + len(rand))
        ax.hist(rand, bins=20, color=BLUE, alpha=0.85, edgecolor="white", linewidth=1)
        ax.axvline(panel, color=ORANGE, lw=2)
        ax.annotate(f"telomere panel {panel:.3f}\nempirical p={p:.2f}", (panel, ax.get_ylim()[1]), xytext=(4, -4),
                    textcoords="offset points", va="top", fontsize=7, color=INK)
        ax.set_xlabel("Within-cancer AUROC")
        ax.set_ylabel(f"Random gene sets (n={len(rand)})")
        ax.set_title(TITLE[t], fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "figS3_random_gene_sets")


def figS_stability(top=30):
    tasks = [t for t in ("alt", "tel") if (C.RESULTS / f"{t}_signature_genes.csv").exists()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 0.17 * top + 1.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = csv(f"{t}_signature_genes.csv").head(top).iloc[::-1]
        colors = [ORANGE if p else BLUE for p in g.in_telomere_panel]
        key = "subsample_selection_frequency" if "subsample_selection_frequency" in g else "selection_frequency"
        ax.barh(np.arange(len(g)), g[key], color=colors, height=0.72)
        ax.axvline(C.SIG_STABLE, color=AXIS, lw=1)
        ax.set_yticks(np.arange(len(g)), g.gene, fontsize=6.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Selection frequency in dedicated stratified subsamples")
        info = json.load(open(C.RESULTS / f"{t}_signature.json"))
        ax.set_title(f"{TITLE[t]}\nmean Jaccard index between outer folds: "
                     f"{info['mean_jaccard_between_folds']:.2f}", fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.0, f"Orange bars: genes in the curated telomere-maintenance panel. Vertical line: "
             f"{C.SIG_STABLE:.0%} selection-frequency threshold.", fontsize=6.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "figS4_signature_stability")


def fig_karyotype_controls():
    tasks = [t for t in ("alt", "tel") if (C.RESULTS / f"{t}_confounding_summary.csv").exists()]
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 3.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        s = pd.read_csv(C.RESULTS / f"{task}_confounding_summary.csv").set_index("model")
        order = ["Cancer type only", "Cancer + molecular subtype only", "CNV only",
                 "CNV only, label loci excluded", "Expression only"]
        order += [m for m in s.index if m.startswith("Expression excluding")]
        order += ["Expression residualized for CNV", "Expression + CNV"]
        colors = {"Cancer type only": GRAY, "Cancer + molecular subtype only": GRAY,
                  "CNV only": BLUE, "CNV only, label loci excluded": BLUE,
                  "Expression only": ORANGE, "Expression residualized for CNV": ORANGE,
                  "Expression + CNV": AQUA}
        rows = [(m, s.loc[m], colors.get(m, ORANGE)) for m in order if m in s.index]
        dot_ci(ax, rows)
        ax.set_xlabel("Within-cancer AUROC (95% CI)")
        ax.set_title(TITLE[task], fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig8_karyotype_controls")


def figS_enrichment(top=10):
    e = csv("validation_enrichment.csv")
    if e is None or e.empty:
        return
    tasks = [t for t in ("alt", "tel") if (e.task == t).any()]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.4 * len(tasks), 0.3 * top + 1.2), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        d = e[e.task == t].nsmallest(top, "p_adj").iloc[::-1]
        ax.barh(np.arange(len(d)), -np.log10(d.p_adj), color=BLUE, height=0.7)
        ax.set_yticks(np.arange(len(d)), [f"{x[:48]} ({s})" for x, s in zip(d.term, d.source)], fontsize=6.5)
        ax.set_xlabel("−log10 adjusted p (g:Profiler, g:SCS)")
        ax.set_title(TITLE[t], fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "figS5_enrichment")


def figS_wu2025_alt():
    r = csv("validation_wu2025_celllines.csv")
    if r is None or r.empty:
        return
    r = r[(r.task == "alt") & (r.scope == "all")]
    panels = [("AUROC", "Unambiguous ALT/ALT-Low versus TEL", 0, 1),
              ("Spearman", "Correlation with C-circle abundance", -1, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.5))
    for ax, (metric, title, xmin, xmax) in zip(axes, panels):
        q = r[r.metric == metric].sort_values("estimate").reset_index(drop=True)
        y = np.arange(len(q))
        ax.errorbar(q.estimate, y, xerr=[q.estimate - q.lo, q.hi - q.estimate], fmt="o", ms=5,
                    color=ORANGE, ecolor=INK2, elinewidth=0.8, capsize=2)
        ax.axvline(0.5 if metric == "AUROC" else 0, color=AXIS, lw=1)
        ax.set_xlim(xmin, xmax)
        ax.set_yticks(y, [disp(m) for m in q.model])
        ax.set_xlabel(f"{metric} (95% CI)")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Locked ALT signatures in the Wu 2025 direct-assay atlas", fontsize=8.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "figS6_wu2025_alt")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def latex_tables():
    lines = []
    key_models = ["Cancer + glioma subtype only", "Barthel 2017 score (published)", "LR (Barthel 2017 genes)",
                  "LR (EXTEND gene set, TERT removed)", "LR (panel)", "Elastic net (panel)",
                  "Linear SVM (panel)", "RF (panel)", "GBM (panel)", "MLP tuned (panel)", SIG, TW,
                  "Elastic net (whole transcriptome)", "Linear SVM (whole transcriptome)"]
    for t in TITLE:
        s = summary(t)
        if s is None:
            continue
        clf = "within_AUROC" in s.columns
        keys = ["within_AUROC", "AUROC", "AUPRC"] if clf else ["within_Spearman", "Spearman", "R2"]
        head = ["Within-cancer AUROC", "Pooled AUROC", "AUPRC"] if clf else ["Within-cancer $\\rho$", "Pooled $\\rho$", "$R^2$"]
        visible_title = f"{TITLE[t]} ($n={int(s.n.iloc[0])}$)"
        header = "Model & " + " & ".join(head) + " \\\\"
        lines += [f"% ---- {t}: {TITLE[t]} (n={int(s.n.iloc[0])})",
                  "\\begin{longtable}{@{}p{0.36\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}p{0.17\\textwidth}@{}}",
                  f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{visible_title}}}}} \\\\", "\\toprule",
                  header, "\\midrule", "\\endfirsthead",
                  f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{{visible_title} (continued)}}}} \\\\", "\\toprule",
                  header, "\\midrule", "\\endhead",
                  "\\midrule \\multicolumn{4}{r@{}}{Continued on next page} \\\\", "\\endfoot",
                  "\\botrule", "\\endlastfoot"]
        order = [m for m in key_models if m in s.index] + [m for m in s.index if m not in key_models]
        for m in order:
            r = s.loc[m]
            cells = []
            for k in keys:
                mark = "$^{\\dagger}$" if m == "Cancer type only" and k.startswith("within") else ""
                cells.append(f"{r[k]:.3f}{mark} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]")
            lines.append(f"{disp(m)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\end{longtable}", ""]
    pairs = [csv(f"{t}_pairs.csv") for t in TITLE]
    pairs = [p for p in pairs if p is not None and not p.empty]
    if pairs:
        pr = pd.concat(pairs)
        pr = pr[pr.metric.isin(["within_AUROC", "within_Spearman"])]
        lines += ["% ---- paired comparisons (within-cancer; paired bootstrap; TOST margin "
                  f"{C.EQUIV_MARGIN})",
                  "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.19\\textwidth}"
                  ">{\\raggedright\\arraybackslash}p{0.31\\textwidth}p{0.16\\textwidth}p{0.12\\textwidth}"
                  "p{0.06\\textwidth}p{0.07\\textwidth}@{}}",
                  "\\multicolumn{6}{@{}l}{\\textbf{Paired comparisons (within-cancer)}} \\\\", "\\toprule",
                  "Task & Comparison & Mean $\\Delta$ [95\\% CI] & 90\\% CI & $p$ & Equivalent \\\\", "\\midrule", "\\endfirsthead",
                  "\\multicolumn{6}{@{}l}{\\textbf{Paired comparisons (within-cancer; continued)}} \\\\", "\\toprule",
                  "Task & Comparison & Mean $\\Delta$ [95\\% CI] & 90\\% CI & $p$ & Equivalent \\\\", "\\midrule", "\\endhead",
                  "\\midrule \\multicolumn{6}{r@{}}{Continued on next page} \\\\", "\\endfoot",
                  "\\botrule", "\\endlastfoot"]
        p_min = 2 / C.N_BOOT                   # smallest non-zero two-sided bootstrap p value
        for _, r in pr.iterrows():
            p = f"$<${p_min:.3f}" if r.p < p_min else f"{r.p:.3f}"
            eq = "yes" if str(r.get("equivalent", "")) == "True" else "no"
            lines.append(f"{TITLE.get(r.task, r.task)} & {disp(r.model_a)} vs {disp(r.model_b)} & "
                         f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] & [{r.lo90:+.3f}, {r.hi90:+.3f}] & "
                         f"{p} & {eq} \\\\")
        lines += ["\\end{longtable}", ""]
    text = "\n".join(lines).replace("_", "\\_").replace("\\\\_", "\\_").replace("%\\_", "%_")
    (C.RESULTS / "tables.tex").write_text(text, encoding="utf-8")


def main():
    # Write the compact numerical table before opening the Matplotlib backends.  On Windows, some PDF-indexing
    # services briefly lock nearby text files after a batch of figure writes.
    latex_tables()
    fig_study_design()
    fig_feature_sets()
    fig_signature_curve()
    fig_subtype_control()
    fig_validation()
    fig_cell_lines()
    fig_survival()
    figS_all_models()
    figS_per_cancer()
    figS_random_null()
    figS_stability()
    figS_enrichment()
    figS_wu2025_alt()
    fig_karyotype_controls()
    print("figures ->", C.FIGURES, "| tables ->", C.RESULTS / "tables.tex")


if __name__ == "__main__":
    main()
