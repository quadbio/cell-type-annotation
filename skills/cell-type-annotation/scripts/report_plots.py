"""Plot generation for the per-level HTML report (build_report.py).

Separated from build_report.py so the report module stays focused on HTML/template
assembly while the matplotlib/scanpy plotting lives here. Imported by build_report.py;
not a standalone entry point. Heavy plotting libs are imported lazily inside the
functions so importing this module stays cheap.

`make_plots()` returns a list of (heading, caption, png_path) tuples that build_report
embeds as base64 figures. It is best-effort: any single plot that fails is skipped with
a note, and a total failure degrades to a text-only report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cta_io import eprint, label_color_map, split_csv


def plot_umap_numbered(adata, viz_key, label_key, embedding_key, out_path, title):
    """UMAP with numeric labels overlaid on cluster centroids + a numbered legend column,
    colored by a similarity-ordered palette (similar cell types -> similar colors).
    Returns the ordered [(number, label, hexcolor)] for an HTML legend too."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.colors import to_hex

    emb = np.asarray(adata.obsm[viz_key])[:, :2]
    cat = adata.obs[label_key].astype("category")
    cats = list(cat.cat.categories)
    codes = cat.cat.codes.values
    N = len(cats)
    if N == 0:
        return []
    cen2d = np.vstack([np.median(emb[codes == i], axis=0) if (codes == i).any()
                       else [np.nan, np.nan] for i in range(N)])
    rep = np.asarray(adata.obsm.get(embedding_key, emb))
    cenHD = np.vstack([np.median(rep[codes == i], axis=0) if (codes == i).any()
                       else np.zeros(rep.shape[1]) for i in range(N)])
    # Similarity order + palette (Unknown -> grey) — SAME helper write_back.py uses to
    # colour adata.uns, so the saved object matches this report.
    order, color_of = label_color_map(cats, cenHD)
    num_of = {ci: pos + 1 for pos, ci in enumerate(order)}

    # size the figure to the data's true aspect so the embedding is not stretched
    dw = float(np.ptp(emb[:, 0])) or 1.0
    dh = float(np.ptp(emb[:, 1])) or 1.0
    scatter_w = 7.0
    scatter_h = max(2.5, scatter_w * dh / dw)
    # Wrap the legend into multiple columns once it gets long, so a 60+ entry legend
    # stays readable (and doesn't force a very tall figure). ~28 rows per column.
    ROWS_PER_COL = 28
    n_cols = max(1, int(np.ceil(N / ROWS_PER_COL)))
    rows_per_col = int(np.ceil(N / n_cols))
    legend_h = 0.26 * rows_per_col + 0.6
    leg_w = 3.0 * n_cols
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(scatter_w + leg_w, max(scatter_h, legend_h)),
        gridspec_kw={"width_ratios": [scatter_w, leg_w]})
    ax.scatter(emb[:, 0], emb[:, 1], c=[color_of[c] for c in codes], s=2.2,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal")                    # <- preserve UMAP geometry, no stretch
    for i in range(N):
        if np.isnan(cen2d[i, 0]):
            continue
        ax.text(cen2d[i, 0], cen2d[i, 1], str(num_of[i]), fontsize=8.5, weight="bold",
                ha="center", va="center", color="black",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.axis("off")
    # numbered legend column
    axl.axis("off")
    legend = []
    # max label chars scale down a little as columns multiply (less width each)
    max_chars = 42 if n_cols == 1 else (30 if n_cols == 2 else 22)
    col_w = 1.0 / n_cols
    for pos, ci in enumerate(order):
        col = pos // rows_per_col
        row = pos % rows_per_col
        x0 = col * col_w
        y = 1 - (row + 0.5) / rows_per_col
        axl.scatter(x0 + 0.02 * col_w + 0.01, y, s=60, marker="s", color=color_of[ci],
                    transform=axl.transAxes, clip_on=False)
        lab = cats[ci]
        txt = lab if len(lab) <= max_chars else lab[:max_chars - 1] + "…"
        axl.text(x0 + 0.06 * col_w + 0.03, y, f"{pos+1}. {txt}", va="center", ha="left",
                 fontsize=7.0 if n_cols > 1 else 7.5, transform=axl.transAxes)
        legend.append((pos + 1, lab, to_hex(color_of[ci])))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.02)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return legend


def make_plots(*, adata_path, cluster_key, viz_key, embedding_key, lognorm_layer,
               split_by, level, labels: pd.DataFrame, wd: Path) -> list[tuple[str, str, Path]]:
    """Best-effort plots. Returns [(heading, caption, png_path)] — heading is a short
    title, caption is the figure legend. Empty on any failure / no adata."""
    if not adata_path or not cluster_key:
        return []
    out: list[tuple[str, str, Path]] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc
        from cta_io import load_adata, resolve_expression
        plots_dir = wd / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        adata = load_adata(adata_path)
        if cluster_key not in adata.obs:
            eprint(f"   plots: cluster key '{cluster_key}' not in obs — skipping plots")
            return []
        if lognorm_layer:
            adata.X = resolve_expression(adata, lognorm_layer)
        # map cluster -> label
        if "label" in labels.columns:
            mapping = {str(r.cluster): str(r.label) for r in labels.itertuples()}
            adata.obs["_label"] = adata.obs[cluster_key].astype(str).map(mapping).astype("category")

        # 1) UMAP with numeric overlays + numbered legend + similarity-ordered palette
        if viz_key and viz_key in adata.obsm and "_label" in adata.obs:
            try:
                p = plots_dir / f"umap_{level}.png"
                plot_umap_numbered(adata, viz_key, "_label",
                                   embedding_key or viz_key, p,
                                   title=f"{level} — assigned label")
                out.append((
                    "Cell map",
                    "Each point is a cell, positioned by the chosen 2-D embedding and "
                    "coloured by its assigned label. The number at each label's centroid "
                    "matches the legend. Label colours are ordered by similarity (computed "
                    "from the anchor embedding) so related cell types share a hue family; "
                    "Unknown/unassigned labels are always grey.",
                    p))
            except Exception as e:  # noqa: BLE001
                eprint(f"   plots: UMAP skipped ({type(e).__name__}: {e})")

        # 2) dotplot of top markers per label (data-driven from COSG specificity)
        cosg = wd / "tables" / f"cosg_{level}.csv"
        if cosg.exists() and "_label" in adata.obs:
            try:
                cdf = pd.read_csv(cosg)
                cl2lab = {str(r.cluster): str(r.label) for r in labels.itertuples()} \
                    if "label" in labels.columns else {}
                cdf["_label"] = cdf["cluster"].astype(str).map(cl2lab)
                present = set(map(str, adata.var_names))
                markers: dict[str, list[str]] = {}
                for lab, g in cdf.dropna(subset=["_label"]).groupby("_label"):
                    genes = [x for x in g.sort_values("rank")["gene"].astype(str) if x in present]
                    # dedupe preserving order, a few per label
                    seen, keep = set(), []
                    for x in genes:
                        if x not in seen:
                            seen.add(x); keep.append(x)
                        if len(keep) >= 4:
                            break
                    if keep:
                        markers[str(lab)] = keep
                if markers:
                    sc.pl.dotplot(adata, markers, groupby="_label", show=False,
                                  standard_scale="var")
                    p = plots_dir / f"dotplot_{level}.png"
                    plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close("all")
                    out.append((
                        "Marker dot plot",
                        "The four most specific marker genes for each label, chosen "
                        "automatically from the data — not a hand-curated list. For every "
                        "cluster the skill runs a Wilcoxon test of that cluster vs the rest, "
                        "then ranks genes by COSG, a specificity score that rewards genes "
                        "which are both strongly expressed in the cluster and largely absent "
                        "elsewhere; the top hits are shown here. Dot colour = the gene's mean "
                        "expression in that label, scaled 0–1 across labels (so it shows "
                        "where each gene peaks); dot size = the fraction of cells in the "
                        "label that express the gene.",
                        p))
            except Exception as e:  # noqa: BLE001
                eprint(f"   plots: dotplot skipped ({type(e).__name__}: {e})")

        # 3) OPTIONAL composition panel(s), one per user-named categorical obs.
        #    Reuses the covariate's ORIGINAL colors from adata.uns[<col>_colors] if present.
        for col in split_csv(split_by):
            if col not in adata.obs or "_label" not in adata.obs:
                eprint(f"   plots: --split-by '{col}' not in obs — skipping that panel")
                continue
            try:
                obs_c = adata.obs[col].astype("category")
                ct = pd.crosstab(adata.obs["_label"], obs_c, normalize="index")
                # original colors from uns if available (scanpy stores them category-aligned)
                uns_key = f"{col}_colors"
                colors = None
                if uns_key in adata.uns and len(adata.uns[uns_key]) >= len(obs_c.cat.categories):
                    cmap = dict(zip(obs_c.cat.categories, list(adata.uns[uns_key])))
                    colors = [cmap.get(c) for c in ct.columns]
                n = len(ct)
                ax = ct.plot(kind="bar", stacked=True, color=colors, width=0.85,
                             figsize=(max(6.0, 0.32 * n), 4.4))
                ax.set_ylabel(f"proportion by {col}"); ax.set_xlabel("")
                ax.set_ylim(0, 1)
                ax.margins(x=0.01)
                ax.tick_params(axis="x", labelrotation=90, labelsize=7)
                ax.legend(title=col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
                p = plots_dir / f"composition_{level}_{col}.png"
                plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close("all")
                out.append((
                    f"Composition by {col}",
                    f"For each label, the proportion of its cells falling in each "
                    f"{col} category.",
                    p))
            except Exception as e:  # noqa: BLE001
                eprint(f"   plots: composition '{col}' skipped ({type(e).__name__}: {e})")
    except Exception as e:  # noqa: BLE001
        eprint(f"   plots degraded ({type(e).__name__}: {e}) — report will be text-only")
    return out
