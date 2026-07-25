"""Inference: ask a trained TxGNN model which drugs it predicts for a disease.

Two things for a query:
  1. a *disease index* (x_idx) - TxGNN speaks indices, not names, so we build a
     name -> idx lookup from TxData.retrieve_id_mapping()
  2. a trained model - predictions from an untrained model are noise

Usage:
    # train once (fast, on the mini KG), then query
    python query_txgnn.py --train --disease "asthma"
    python query_txgnn.py --disease "asthma" --relation indication --topk 15
    python query_txgnn.py --list-diseases asthma      # find what's queryable
"""
import argparse
import os

from txgnn import TxData, TxGNN, TxEval

CKPT = "./model_ckpt_mini"


def build_name_lookup(txdata):
    """name (lowercased) -> disease idx, via idx -> id -> name."""
    m = txdata.retrieve_id_mapping()
    idx2id, id2name = m["idx2id_disease"], m["id2name_disease"]
    name2idx = {}
    for idx, did in idx2id.items():
        nm = id2name.get(did)
        if nm:
            name2idx.setdefault(str(nm).lower(), idx)
    return name2idx, idx2id, id2name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data_mini")
    ap.add_argument("--disease", help="disease name (substring match)")
    ap.add_argument("--disease-idx", type=float, help="exact disease idx, skips name lookup")
    ap.add_argument("--relation", default="indication",
                    choices=["indication", "contraindication", "off-label use"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--train", action="store_true", help="train and save a checkpoint first")
    ap.add_argument("--list-diseases", metavar="SUBSTR", help="list matching disease names and exit")
    args = ap.parse_args()

    # fail fast, before the (slow) KG load, if there is nothing to query
    if not args.list_diseases and args.disease is None and args.disease_idx is None:
        ap.error("nothing to query. Pass --disease NAME, --disease-idx IDX, "
                 "or --list-diseases SUBSTR to see what is available.")

    txdata = TxData(data_folder_path=args.data)
    txdata.prepare_split(split="complex_disease", seed=42)

    name2idx, _, _ = build_name_lookup(txdata)

    if args.list_diseases:
        q = args.list_diseases.lower()
        hits = [(n, i) for n, i in name2idx.items() if q in n]
        print(f"{len(hits)} disease(s) matching {args.list_diseases!r}:")
        for n, i in sorted(hits)[:40]:
            print(f"  idx={i:<12} {n}")
        return

    model = TxGNN(data=txdata, weight_bias_track=False,
                  proj_name="TxGNN_query", exp_name="TxGNN_query", device="cpu")

    if args.train or not os.path.exists(CKPT):
        print("Training a small model (demo settings, not paper quality)...")
        model.model_initialize(n_hid=32, n_inp=32, n_out=32, proto=True, proto_num=3,
                               attention=False, sim_measure="all_nodes_profile",
                               agg_measure="rarity", num_walks=10, path_length=2)
        model.pretrain(n_epoch=1, learning_rate=1e-3, batch_size=1024, train_print_per_n=500)
        model.finetune(n_epoch=2, learning_rate=5e-4, train_print_per_n=1, valid_per_n=1000)
        model.save_model(CKPT)
        print(f"saved -> {CKPT}")
    else:
        print(f"Loading checkpoint {CKPT} ...")
        model.load_pretrained(CKPT)

    # ---- resolve the query disease --------------------------------------------
    if args.disease_idx is not None:
        didx = args.disease_idx
        label = f"idx={didx}"
    else:
        q = args.disease.lower()
        exact = name2idx.get(q)
        if exact is not None:
            didx, label = exact, args.disease
        else:
            hits = [(n, i) for n, i in name2idx.items() if q in n]
            if not hits:
                raise SystemExit(f"No disease matching {args.disease!r}. "
                                 f"Try: python query_txgnn.py --list-diseases {args.disease}")
            label, didx = sorted(hits)[0]
            print(f"(matched {len(hits)} names, using {label!r})")

    # ---- run inference ---------------------------------------------------------
    print(f"\nQuerying '{label}' (idx {didx}) for relation '{args.relation}' ...\n")
    result = TxEval(model=model).eval_disease_centric(
        disease_idxs=[didx], relation=args.relation,
        save_result=False, show_plot=False, verbose=False)

    # eval_disease_centric returns a DataFrame when `relation` is given, and a
    # dict {rev_<relation>: DataFrame} when it is None.
    if isinstance(result, dict):
        result = result["rev_" + args.relation]

    if len(result) == 0:
        raise SystemExit(
            f"No predictions returned for idx {didx}. That disease is probably absent "
            f"from this split's test set. Use --list-diseases to find a valid one, or "
            f"train with --split full_graph for deployment-style queries.")

    ranked = result["Ranked List"]
    # rows are keyed by disease ID, not the x_idx passed in, so fall back to the first
    # row (we only ever query one disease at a time)
    entry = ranked[didx] if didx in ranked.index else ranked.iloc[0]

    print(f"Top {args.topk} predicted drugs ({args.relation}):")
    for r, drug in enumerate(entry[:args.topk], 1):
        print(f"  {r:2}. {drug}")


if __name__ == "__main__":
    main()
