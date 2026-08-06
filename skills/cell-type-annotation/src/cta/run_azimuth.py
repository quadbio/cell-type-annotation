#!/usr/bin/env python
"""Cross-check via pan-human Azimuth — a HYPOTHESIS (not a verdict). HUMAN-only.

Pan-human Azimuth (Satija lab / NIH HuBMAP; Sarkar et al., bioRxiv 2026,
doi:10.64898/2026.07.16.738997) is a SUPERVISED neural-network classifier — not a foundation
model — over a fixed 5,055-gene HGNC panel, with 8 chained hierarchical softmax heads. It was
trained on ~9.7M curated cells across 23 HUMAN tissues, plus ~145k deliberate negative examples
(empty droplets + simulated high-UMI mixtures) labelled `Unassigned`.

SCOPE, from the authors: the model "was not designed to capture developmental stages,
disease-specific states, or cancer programs", though non-cancer disease usually still maps
sensibly onto the nearest healthy identity. So: adult/postnatal HUMAN tissue is its home turf;
fetal/organoid and tumour are out of scope; non-human is not supported at all.

DISQUALIFIED ON TARGETED PANELS. panhumanpy silently zero-fills panel genes the query lacks and
only warns below 20% overlap, so a 300-plex MERSCOPE/Xenium/CosMx panel (~5% of 5,055) yields
confident garbage. The paper's only spatial result is whole-transcriptome VisiumHD **with cell
segmentation** — and even there, fixed 8um bins pushed most spots to `Unassigned`. This command
refuses below --min-panel-overlap.

It CANNOT be constrained to a tissue: it always predicts over the full pan-body label space.
That is why a well-matched tissue-specific reference still outranks it (reference/method_selection.md).

Cost: CPU, ~50-60k cells/min; 84 MB model (provision once with `cta reference fetch-azimuth`);
peak RAM roughly 2x the count matrix (panhumanpy casts X to float64), plus a retained
(n_cells x 2117) softmax buffer. A GPU buys nothing below ~10M cells.

THREE GRANULARITIES, ONE MODEL. `--granularity broad|medium|fine` (13 / 85 / 291 classes) selects
which refined level to report. Run it up to three times with distinct `--label-col` values to get
one `refmap_*` table per level — the cards/report auto-discover each.

CANONICAL SHAPE. Expects the skill's canonical object (`cta data canonicalize`): gene SYMBOLS as
var_names, raw counts in `layers['counts']`, log-norm in `X`, exact log1p-CP10K in
`layers['lognorm_cp10k']`. panhumanpy reads `.X` ONLY (no layer argument) and decides what to do
by sniffing integer-ness — so handing it the canonical `X` (the user's own log-norm, NOT
necessarily CP10K) would silently mis-scale. Hence only two `--counts-layer` values are allowed.

Writes tables/refmap_<level>_<label-col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
plus two named-differently side tables (NOT auto-discovered as cross-checks):
  tables/azimuth_state_<level>.csv  — per-cluster QC corroboration for the `state` axis
  tables/azimuth_cl_<level>.csv     — per-cluster Cell Ontology id + SKOS match strength
"""

from __future__ import annotations

import numpy as np
import typer

from cta.io import (
    CLUSTER_KEY_OPT,
    LEVEL_OPT,
    WORKDIR_OPT,
    _looks_like_ensembl,
    aggregate_predictions,
    eprint,
    load_adata,
    require_obs,
    resolve_workdir,
    safe_cluster_id,
    save_table,
)

GRANULARITIES = ("broad", "medium", "fine")
#: panhumanpy writes this literal STRING into azimuth_broad/medium/fine when a cell's 8 heads
#: do not form a self-consistent path (upstream issue #47). It is not a label; left in place it
#: can win a cluster's majority vote. We fall back to `final_level_labels` for those cells.
INCONSISTENT = "False"
#: 'cell' — the crosswalk target for `Doublet like cell`. Syntactically a CL id, semantically
#: no assertion at all, so it must not reach a labels CSV.
USELESS_CL = {"CL:0000000"}
#: Only these two canonical layers are safe to hand panhumanpy (see the module docstring).
ALLOWED_LAYERS = ("counts", "lognorm_cp10k")


def _leaf(label) -> str:
    """Leaf of an Azimuth label ('A|B|C' -> 'C'). Most refined labels are already leaves."""
    return str(label).rsplit("|", 1)[-1]


def _cl_crosswalk(model_version: str) -> tuple[dict, dict]:
    """label -> CL_ID and label -> SKOS match, from panhumanpy's shipped ontology map.

    We do NOT use panhumanpy's own `map_to_cell_ontology`: it does one exact lookup, but a few
    refined labels are short pipe-paths rather than bare leaves, so `_cl_for` tries the label
    as-is and then its leaf. That resolves the large majority of observed labels; `Unassigned`
    correctly has no CL term.
    """
    from importlib.resources import files

    import pandas as pd

    base = files(f"panhumanpy._tools.{model_version}")
    om = pd.read_csv(str(base / "cell_ontology_map" / "cell_ontology_map.csv"))
    lab = om["Annotation_Label"].astype(str)
    return (
        dict(zip(lab, om["CL_ID"].astype(str), strict=True)),
        dict(zip(lab, om["CL_Match"].astype(str), strict=True)),
    )


def _cl_for(label: str, ids: dict, match: dict) -> tuple[str, str]:
    """(CL_ID, SKOS match) for one label; ('', '') when there is no usable term."""
    for key in (str(label), _leaf(label)):
        cl = ids.get(key)
        if cl and cl not in USELESS_CL:
            return cl, match.get(key, "")
    return "", ""


def _azimuth_input(adata, counts_layer: str):
    """(matrix, normalization_override) for panhumanpy — or a hard stop.

    `counts` -> raw counts, panhumanpy computes exact CP10K+log1p itself (override False).
    `lognorm_cp10k` -> the canonical precomputed exact CP10K+log1p (override True, so
    panhumanpy leaves it alone rather than re-deriving it).
    Anything else is refused: panhumanpy would silently treat it as already-normalized.
    """
    if counts_layer not in ALLOWED_LAYERS:
        raise SystemExit(
            f"ERROR: --counts-layer must be one of {ALLOWED_LAYERS}, got {counts_layer!r}.\n"
            "panhumanpy reads .X only and infers normalization by sniffing integer-ness, so an "
            "arbitrary layer (including the canonical log-norm `X`, which is the user's own "
            "normalization and NOT necessarily CP10K) would be silently mis-scaled. Run "
            "`cta data canonicalize` — it writes both allowed layers."
        )
    mat = adata.layers.get(counts_layer)
    if mat is None:
        raise SystemExit(
            f"ERROR: layer '{counts_layer}' not in adata.layers {list(adata.layers.keys())}. "
            "Run `cta data canonicalize` so raw counts land in layers['counts'] and the exact "
            "log1p-CP10K in layers['lognorm_cp10k'] (and var_names become gene symbols)."
        )
    return mat, counts_layer == "lognorm_cp10k"


def _state_table(clusters, meta, broad_labels):
    """Per-cluster QC corroboration for the skill's `state` axis (NOT a cross-check table)."""
    import pandas as pd

    raw = meta["azimuth_fine"].astype(str).to_numpy()
    df = pd.DataFrame(
        {
            "cluster": np.asarray(clusters, dtype=str),
            "unassigned": (np.asarray(broad_labels, dtype=str) == "Unassigned"),
            "doublet_like": (np.asarray(broad_labels, dtype=str) == "Doublet like cell"),
            "inconsistent": (raw == INCONSISTENT),
            "conf": np.asarray(meta["final_level_confidence"], dtype=float),
        }
    )
    rows = []
    for cl, g in df.groupby("cluster", sort=True):
        rows.append(
            {
                "cluster": cl,
                "n_cells": int(len(g)),
                "frac_unassigned": round(float(g["unassigned"].mean()), 4),
                "frac_doublet_like": round(float(g["doublet_like"].mean()), 4),
                "frac_inconsistent_hierarchy": round(float(g["inconsistent"].mean()), 4),
                "mean_conf": round(float(np.nanmean(g["conf"].to_numpy())), 4),
            }
        )
    return pd.DataFrame(rows)


def cli(
    query: str = typer.Option(
        ..., "--query", "--adata", help="the query AnnData to annotate (alias: --adata)"
    ),
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    granularity: str = typer.Option(
        "fine", help="which refined level to report: broad (13) | medium (85) | fine (291)"
    ),
    counts_layer: str = typer.Option(
        "counts",
        help="canonical layer to feed panhumanpy: 'counts' (raw; it normalizes) or "
        "'lognorm_cp10k' (exact log1p-CP10K; passed through). Nothing else is accepted.",
    ),
    label_col: str = typer.Option(
        "azimuth", help="name used in the output table filename (use one per granularity)"
    ),
    model_version: str = typer.Option("v1", help="v1 (calibrated + Unassigned class) or v0"),
    eval_batch_size: int = typer.Option(8192, help="cells per inference minibatch"),
    feature_names_col: str = typer.Option(
        None, help="var column holding gene symbols, if var_names are not symbols"
    ),
    min_panel_overlap: float = typer.Option(
        20.0,
        help="refuse below this %% of the 5,055-gene panel — targeted spatial panels are "
        "zero-filled into confident garbage. Pass 0 to override deliberately.",
    ),
) -> None:
    """Cross-check via pan-human Azimuth — a hypothesis (not a verdict). HUMAN-only, adult."""
    if granularity not in GRANULARITIES:
        raise SystemExit(
            f"ERROR: --granularity must be one of {GRANULARITIES}, got {granularity!r}"
        )
    try:
        import panhumanpy
    except ImportError:
        raise SystemExit(
            "ERROR: panhumanpy is not installed. It lives in the skill's `azimuth` pixi env "
            "(it pulls TensorFlow, kept out of the lean CPU default):\n"
            "    pixi install --frozen -e azimuth\n"
            "    pixi run --frozen -e azimuth cta reference fetch-azimuth   # 84 MB, once, needs network\n"
            "    pixi run --frozen -e azimuth cta crosscheck azimuth ...\n"
            "Or, since `cta` declares no dependencies, install it alongside panhumanpy in any "
            "Python 3.9-3.12 env of your own and run `cta` from there."
        ) from None

    import anndata as ad

    eprint(f">> loading query {query}")
    query_obj = load_adata(query)
    require_obs(query_obj, cluster_key, "cluster key")

    if feature_names_col is None and _looks_like_ensembl(query_obj.var_names):
        raise SystemExit(
            "ERROR: var_names look like Ensembl IDs, but pan-human Azimuth matches on HGNC "
            "symbols. panhumanpy does NOT fail on this — it zero-fills every panel gene and "
            "returns confident nonsense (upstream issue #39). Run `cta data canonicalize` so "
            "var_names become symbols, or pass --feature-names-col <var column>."
        )

    mat, override = _azimuth_input(query_obj, counts_layer)

    # Hand panhumanpy a minimal object: it only ever reads .X and var_names, and building a
    # slim copy keeps the float64 cast off the full annotated object.
    tmp = ad.AnnData(X=mat, obs=query_obj.obs[[cluster_key]].copy(), var=query_obj.var[[]].copy())
    tmp.var_names = query_obj.var_names

    from importlib.resources import files

    panel = set(
        (
            files(f"panhumanpy._tools.{model_version}")
            / "inference_feature_panel"
            / "inference_feature_panel.txt"
        )
        .read_text()
        .split()
    )
    overlap = len(set(tmp.var_names.astype(str)) & panel)
    pct = 100.0 * overlap / len(panel)
    eprint(f">> panel overlap: {overlap}/{len(panel)} ({pct:.1f}%)")
    if pct < min_panel_overlap:
        raise SystemExit(
            f"ERROR: only {pct:.1f}% of Azimuth's {len(panel)}-gene panel is present "
            f"(< --min-panel-overlap {min_panel_overlap}%). panhumanpy zero-fills the missing "
            "genes and will still return confident labels — which is exactly the failure mode "
            "this gate exists to prevent. This is the expected outcome on a targeted spatial "
            "panel (MERSCOPE/Xenium/CosMx): use `cta crosscheck reference-map` (CellMapper "
            "cross-modality CCA) instead. Pass --min-panel-overlap 0 only to override knowingly."
        )

    eprint(f">> running pan-human Azimuth ({model_version}, {granularity}) on {tmp.n_obs} cells")
    az = panhumanpy.AzimuthNN(
        tmp,
        feature_names_col=feature_names_col,
        model_version=model_version,
        eval_batch_size=eval_batch_size,
        normalization_override=override,
        output_mode="minimal",
        refine=True,
    )
    meta = az.cells_meta

    def _resolve(g: str):
        lab = meta[f"azimuth_{g}"].astype(str).to_numpy().copy()
        bad = lab == INCONSISTENT
        if bad.any() and "final_level_labels" in meta.columns:
            lab[bad] = meta["final_level_labels"].astype(str).to_numpy()[bad]
        return np.array([_leaf(x) for x in lab], dtype=object)

    labels = _resolve(granularity)
    conf = np.asarray(meta["final_level_confidence"], dtype=float)
    clusters = query_obj.obs[cluster_key].astype(str).to_numpy()

    wd = resolve_workdir(workdir)
    out = aggregate_predictions(clusters, labels, conf, f"azimuth_{granularity}")

    # Cell Ontology ids for each cluster's winning label — an INDEPENDENT assertion to
    # reconcile against the cl_id you set in labels/annotation_<level>.csv, not a substitute
    # for the knowledgebase-MCP validation at write-back.
    ids, match = _cl_crosswalk(model_version)
    import pandas as pd

    cl_rows = [
        dict(
            zip(
                ("cluster", "top1", "cl_id", "cl_match"),
                (r.cluster, r.top1, *_cl_for(r.top1, ids, match)),
                strict=True,
            )
        )
        for r in out.itertuples()
    ]
    save_table(pd.DataFrame(cl_rows), wd, f"azimuth_cl_{level}.csv")
    save_table(_state_table(clusters, meta, _resolve("broad")), wd, f"azimuth_state_{level}.csv")
    save_table(out, wd, f"refmap_{level}_{safe_cluster_id(label_col)}.csv")

    n_cl = sum(1 for r in cl_rows if r["cl_id"])
    eprint(
        f">> done. {len(out)} clusters, {n_cl} with a CL id "
        f"(supervised HUMAN model — treat as a hypothesis)."
    )
    eprint(
        f">> state evidence in azimuth_state_{level}.csv: read frac_unassigned (low-quality OR "
        f"mixture), frac_doublet_like, frac_inconsistent_hierarchy against `cta cluster profile`."
    )
