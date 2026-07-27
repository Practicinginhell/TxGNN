"""
Build a small, self-consistent subset of the TxGNN knowledge graph for fast testing.

Why this exists: the full KG is 4.05M edges stored as 8.1M rows in a 982 MB kg.csv, so
a single end-to-end run takes hours. This script carves out a connectivity-aware
subgraph that keeps the same column schema, so the full pipeline (preprocess -> split
-> DGL graph -> pretrain -> finetune -> evaluate) exercises the same code paths in
~minutes.

The subset keeps all 10 node types and all 30 relations. Five of those types (anatomy,
pathway, biological_process, cellular_component and molecular_function) attach only
through gene/protein and exposure, never directly to a drug or a disease, so the one-hop
expansion from the seed diseases misses them completely. Step 3b pulls in a bounded
sample of each, sized by --max-new-per-type. Pass 0 for the older one-hop behaviour,
which yields 5 node types and 16 relations.

The graph is still far sparser than the full KG, so it is for smoke testing rather than
for measuring anything.

Design constraints discovered from txgnn/utils.py:
  * preprocess_kg() de-duplicates each relation to ONE orientation via
    `d_off[d_off.x_type == d_off.x_type.iloc[0]]`, so every kept edge must be present
    in BOTH directions, exactly like the raw kg.csv.
  * complex_disease_fold() splits *diseases* with frac [0.83125, 0.11875, 0.05], so we
    need enough treated diseases for the 5% test bucket to be non-empty.
  * Node indices are re-derived per node type inside preprocess_kg(), so x_index /
    y_index only need to be internally consistent.

Usage:
    python make_mini_kg.py --src data_full/kg.csv --out data_mini \
        --n-diseases 400 --max-ppi 20000 --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd

DD_RELS = ["indication", "contraindication", "off-label use"]


def symmetric_keep(df, keep_mask):
    """Keep an edge only if both of its directions survive.

    kg.csv stores every edge twice (u->v and v->u). We build an undirected key and
    keep all rows whose key was selected, which restores the both-directions
    invariant that preprocess_kg() relies on.
    """
    sub = df[keep_mask]
    key = pd.Series(
        [
            "_".join(sorted([a, b])) + "|" + r
            for a, b, r in zip(
                sub.x_index.astype(str), sub.y_index.astype(str), sub.relation
            )
        ],
        index=sub.index,
    )
    kept_keys = set(key.unique())

    full_key = pd.Series(
        [
            "_".join(sorted([a, b])) + "|" + r
            for a, b, r in zip(
                df.x_index.astype(str), df.y_index.astype(str), df.relation
            )
        ],
        index=df.index,
    )
    return df[full_key.isin(kept_keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_full/kg.csv")
    ap.add_argument("--out", default="data_mini")
    ap.add_argument("--n-diseases", type=int, default=400,
                    help="number of treated diseases to seed the subgraph with")
    # Caps count rows, not edges, and are applied BEFORE the both-directions repair in
    # step 6, so the final per-relation row count comes out at roughly 2x the cap once
    # mirror rows are added back. 20000 yields about 37k-40k rows for that relation.
    ap.add_argument("--max-ppi", type=int, default=20000,
                    help="cap on protein_protein rows before mirroring (final count is ~2x)")
    ap.add_argument("--max-per-rel", type=int, default=20000,
                    help="cap on any other single relation before mirroring (final count is ~2x)")
    # anatomy, pathway and the three Gene Ontology types attach only through
    # gene/protein and exposure, never directly to a drug or a disease, so the
    # one-hop expansion in step 3 can never reach them. See step 3b.
    ap.add_argument("--max-new-per-type", type=int, default=300,
                    help="nodes to pull in per node type that sits two hops from the seed "
                         "diseases; 0 keeps the one-hop behaviour and drops those types")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.src} ...")
    kg = pd.read_csv(args.src, low_memory=False)
    print(f"  full KG: {len(kg):,} edges, {kg.relation.nunique()} relations")

    # ---- 1. pick seed diseases that actually have treatments -------------------
    dd = kg[kg.relation.isin(DD_RELS)]
    # diseases appear on whichever side has y_type == 'disease'
    dis_counts = dd[dd.y_type == "disease"].groupby("y_index").size()
    # prefer well-connected diseases so the subgraph is dense enough to learn on
    eligible = dis_counts[dis_counts >= 3].index.values
    print(f"  diseases with >=3 treatments: {len(eligible):,}")

    n = min(args.n_diseases, len(eligible))
    seed_diseases = set(rng.choice(eligible, size=n, replace=False).tolist())
    print(f"  seeded with {len(seed_diseases)} diseases")

    # ---- 2. drugs attached to those diseases ----------------------------------
    dd_keep = dd[dd.y_index.isin(seed_diseases) | dd.x_index.isin(seed_diseases)]
    drugs = set(dd_keep[dd_keep.x_type == "drug"].x_index.unique()) | set(
        dd_keep[dd_keep.y_type == "drug"].y_index.unique()
    )
    print(f"  drugs pulled in: {len(drugs):,}")

    # ---- 3. one-hop neighbours of the seed nodes (proteins, phenotypes, ...) ---
    core = seed_diseases | drugs
    touches_core = kg.x_index.isin(core) | kg.y_index.isin(core)
    one_hop = kg[touches_core]
    neighbours = set(one_hop.x_index.unique()) | set(one_hop.y_index.unique())
    print(f"  node set after 1-hop: {len(neighbours):,}")

    # ---- 3b. second hop, for the types that are never adjacent to the core -----
    # Five node types (anatomy, pathway, biological_process, cellular_component,
    # molecular_function) reach the graph only through gene/protein and exposure.
    # A one-hop expansion from diseases and drugs cannot reach them at all, so
    # they would drop out entirely. Pull in a bounded number of each.
    present = set(one_hop.x_type) | set(one_hop.y_type)
    inside_x = kg.x_index.isin(neighbours)
    inside_y = kg.y_index.isin(neighbours)
    crossing = kg[inside_x != inside_y]
    on_x = inside_x[crossing.index].values
    outside = pd.DataFrame({
        "idx": np.where(on_x, crossing.y_index, crossing.x_index),
        "type": np.where(on_x, crossing.y_type, crossing.x_type),
    }).drop_duplicates()

    added, gained = set(), []
    for node_type, grp in outside.groupby("type"):
        if node_type in present or args.max_new_per_type < 1:
            continue
        n = min(args.max_new_per_type, len(grp))
        added.update(rng.choice(grp.idx.values, size=n, replace=False).tolist())
        gained.append(node_type)
    if added:
        neighbours |= added
        print(f"  2-hop pulled in {len(added):,} nodes across {len(gained)} new types: "
              f"{', '.join(gained)}")
        print(f"  node set after 2-hop: {len(neighbours):,}")

    # ---- 4. keep edges whose BOTH endpoints are in the node set ---------------
    both_in = kg.x_index.isin(neighbours) & kg.y_index.isin(neighbours)
    sub = kg[both_in]
    print(f"  rows with both endpoints kept: {len(sub):,}")

    # ---- 5. cap the huge relations so the graph stays small -------------------
    pieces = []
    for rel, grp in sub.groupby("relation"):
        cap = args.max_ppi if rel == "protein_protein" else args.max_per_rel
        if rel in DD_RELS:
            pieces.append(grp)  # never downsample drug-disease edges: they are the task
        elif len(grp) > cap:
            pieces.append(grp.sample(n=cap, random_state=args.seed))
        else:
            pieces.append(grp)
    sub = pd.concat(pieces)

    # ---- 6. restore the both-directions invariant -----------------------------
    mask = kg.index.isin(sub.index)
    sub = symmetric_keep(kg, mask)

    # Drop relations that ended up with a single orientation. preprocess_kg keeps one
    # orientation of a heterogeneous relation via `d_off[d_off.x_type ==
    # d_off.x_type.iloc[0]]`, which matches every row when the mirror is missing, so
    # the de-duplication silently becomes a no-op. A heterogeneous relation holding
    # both orientations has two distinct x_types; a homogeneous one has a single
    # x_type equal to its y_type, and one orientation is normal there.
    ok_rels = []
    for rel, grp in sub.groupby("relation"):
        homogeneous = (grp.x_type.nunique() == 1 and grp.y_type.nunique() == 1
                       and grp.x_type.iloc[0] == grp.y_type.iloc[0])
        if homogeneous or grp.x_type.nunique() == 2:
            ok_rels.append(rel)
    sub = sub[sub.relation.isin(ok_rels)]

    out_path = os.path.join(args.out, "kg.csv")
    sub.to_csv(out_path, index=False)

    # ---- 7. report -------------------------------------------------------------
    dd_final = sub[sub.relation.isin(DD_RELS)]
    n_dis = dd_final[dd_final.y_type == "disease"].y_index.nunique()
    # Rows are not edges: every edge is stored once per direction. Canonicalise each
    # row to an unordered endpoint pair to count the edges behind them.
    lo = np.minimum(sub.x_index.values, sub.y_index.values)
    hi = np.maximum(sub.x_index.values, sub.y_index.values)
    n_edges = len(pd.DataFrame({"rel": sub.relation.values, "lo": lo, "hi": hi})
                  .drop_duplicates())
    # complex_disease_fold cuts the shuffled disease list with np.split at
    # int((frac[0] + frac[1]) * n), so the test bucket gets whatever is left over.
    # Mirror that sum literally: 0.83125 + 0.11875 is 0.9500000000000001, not 0.95.
    n_test = n_dis - int((0.83125 + 0.11875) * n_dis)
    print("\n=== mini KG written ===")
    print(f"  path       : {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  rows       : {len(sub):,}  (from {len(kg):,})")
    print(f"  edges      : {n_edges:,}  (each stored in both directions)")
    print(f"  relations  : {sub.relation.nunique()} (from {kg.relation.nunique()})")
    print(f"  node types : {sorted(set(sub.x_type.unique()) | set(sub.y_type.unique()))}")
    print(f"  treated diseases : {n_dis}  -> {n_test} land in the test split")
    print("\n  rows per relation:")
    for rel, cnt in sub.relation.value_counts().items():
        print(f"    {rel:35s} {cnt:>8,}")

    # node.csv / edges.csv are downloaded by TxData.__init__, write consistent
    # subsets so it finds "local copies" and skips the multi-GB download.
    nodes = pd.concat([
        # str.removeprefix() would read better here, but it is Python 3.9+ and this
        # project is pinned to 3.8 (DGL 0.5.2 has no newer wheels).
        sub[["x_index", "x_id", "x_type", "x_name", "x_source"]].rename(
            columns=lambda c: c[len("x_"):] if c.startswith("x_") else c),
        sub[["y_index", "y_id", "y_type", "y_name", "y_source"]].rename(
            columns=lambda c: c[len("y_"):] if c.startswith("y_") else c),
    ]).drop_duplicates(subset=["index"]).sort_values("index")
    nodes.columns = ["node_index", "node_id", "node_type", "node_name", "node_source"]

    nodes.to_csv(os.path.join(args.out, "node.csv"), sep="\t", index=False)

    sub[["relation", "display_relation", "x_index", "y_index"]].to_csv(
        os.path.join(args.out, "edges.csv"), index=False)
    print(f"\n  also wrote node.csv ({len(nodes):,} nodes) and edges.csv")


if __name__ == "__main__":
    main()
