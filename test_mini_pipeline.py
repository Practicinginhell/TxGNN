"""Full-KG TxGNN run with the paper's hyperparameters.

Parameters match reproduce/train.py (the authors' own training script), not the tiny
smoke-test values this file started with. For the fast sanity-check version, see
make_mini_kg.py + `--data ./data_mini` with the reduced settings below.

  split -> DGL graph -> pretrain -> finetune -> disease-centric evaluation

Runtime warning: the full KG is 8.1M edges. On a GPU expect several hours for 500
finetune epochs; on CPU this is not practical. The first run also downloads ~945 MB
and spends several minutes building kg_directed.csv and the splits (both are cached
afterwards).

Usage:
    python test_mini_pipeline.py                      # paper settings, full KG
    python test_mini_pipeline.py --finetune-epochs 50 # shorter run
    python test_mini_pipeline.py --data ./data_mini --n-hid 32 \
        --pretrain-epochs 1 --finetune-epochs 2 --valid-per-n 1000   # smoke test
"""
import argparse
import os
import time

t0 = time.time()


def step(msg):
    print(f"\n{'='*60}\n[{time.time()-t0:7.1f}s] {msg}\n{'='*60}", flush=True)


p = argparse.ArgumentParser()
p.add_argument('--data', default='./data', help='full KG folder (auto-downloads)')
p.add_argument('--split', default='complex_disease',
               help="'complex_disease' for the paper's zero-shot setting; "
                    "'full_graph' for deployment (trains on everything)")
p.add_argument('--seed', type=int, default=42)
p.add_argument('--device', default=None, help="e.g. 'cuda:0' or 'cpu' (default: auto)")
# paper hyperparameters (reproduce/train.py)
p.add_argument('--n-hid', type=int, default=100)
p.add_argument('--n-inp', type=int, default=100)
p.add_argument('--n-out', type=int, default=100)
p.add_argument('--proto-num', type=int, default=3)
p.add_argument('--num-walks', type=int, default=200)
p.add_argument('--path-length', type=int, default=2)
p.add_argument('--pretrain-epochs', type=int, default=2)
p.add_argument('--pretrain-lr', type=float, default=1e-3)
p.add_argument('--batch-size', type=int, default=1024)
p.add_argument('--finetune-epochs', type=int, default=500)
p.add_argument('--finetune-lr', type=float, default=5e-4)
p.add_argument('--valid-per-n', type=int, default=20,
               help='how often to validate; best_model is the best validation macro-AUROC')
p.add_argument('--save-dir', default='./saved_models/TxGNN_full')
args = p.parse_args()

from txgnn import TxData, TxGNN, TxEval

if args.device is None:
    import torch
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu':
        print('WARNING: no GPU detected. A full-KG run on CPU is impractically slow.')
print(f'device: {args.device}')

step('1. Load KG + build split')
TxDataObj = TxData(data_folder_path=args.data)
TxDataObj.prepare_split(split=args.split, seed=args.seed, no_kg=False)

print("train edges:", len(TxDataObj.df_train))
print("valid edges:", len(TxDataObj.df_valid))
print("test  edges:", len(TxDataObj.df_test))
print("DGL graph:", TxDataObj.G)

step('2. Initialize model')
model = TxGNN(data=TxDataObj,
              weight_bias_track=False,
              proj_name='TxGNN_full',
              exp_name='TxGNN_full',
              device=args.device)

model.model_initialize(n_hid=args.n_hid, n_inp=args.n_inp, n_out=args.n_out,
                       proto=True, proto_num=args.proto_num, attention=False,
                       sim_measure='all_nodes_profile',
                       agg_measure='rarity',
                       num_walks=args.num_walks, path_length=args.path_length)

step(f'3. Pretrain ({args.pretrain_epochs} epochs, link prediction on all edge types)')
model.pretrain(n_epoch=args.pretrain_epochs, learning_rate=args.pretrain_lr,
               batch_size=args.batch_size, train_print_per_n=20)

step(f'4. Finetune ({args.finetune_epochs} epochs, metric learning on drug-disease)')
# best_model = highest validation macro-AUROC, checked every valid_per_n epochs;
# TxEval evaluates that best checkpoint, not the last one.
model.finetune(n_epoch=args.finetune_epochs, learning_rate=args.finetune_lr,
               train_print_per_n=5, valid_per_n=args.valid_per_n)

step('5. Save model')
os.makedirs(os.path.dirname(args.save_dir.rstrip('/')) or '.', exist_ok=True)
model.save_model(args.save_dir)
print('saved ->', args.save_dir)

step('6. Disease-centric evaluation (best checkpoint)')
res = TxEval(model=model).eval_disease_centric(disease_idxs='test_set',
                                               show_plot=False, verbose=True,
                                               save_result=True, return_raw=False,
                                               save_name=os.path.join(args.data, 'TxGNN_full_eval.pkl'))
print(res.head() if hasattr(res, 'head') else res)

step(f'DONE in {time.time()-t0:.1f}s')
