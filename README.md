# TxGNN: Zero-shot prediction of therapeutic use with geometric deep learning and human centered design

This repository hosts the official implementation of TxGNN, a model for identifying therapeutic opportunities for diseases with limited treatment options and minimal molecular understanding that leverages recent advances in geometric deep learning and human-centered.

TxGNN is a graph neural network pre-trained on a comprehensive knowledge graph of 17,080 clinically-recognized diseases and 7,957 therapeutic candidates. The model can process various therapeutic tasks, such as indication and contraindication prediction, in a unified formulation. Once trained, we show that TxGNN can perform zero-shot inference on new diseases without additional parameters or fine-tuning on ground truth labels.

### MedRxiv preprint is at [https://www.medrxiv.org/content/10.1101/2023.03.19.23287458v2](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458v2)

### TxGNN Explorer of model predictions and explanations is at [http://txgnn.org](http://txgnn.org/)

![TxGNN](fig/txgnn_fig1.png)

### Installation

```bash
conda create --name txgnn_env python=3.8
conda activate txgnn_env
# Install PyTorch via https://pytorch.org/ with your CUDA versions
conda install -c dglteam dgl-cuda{$CUDA_VERSION}==0.5.2 # checkout https://www.dgl.ai/pages/start.html for more info, as long as it is DGL 0.5.2
pip install TxGNN
```

Note that if you want to use disease-area split, you should also install PyG following [this instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) since some legacy data processing code uses PyG utility functions.

TxGNN targets a Python 3.8 / DGL 0.5.2 stack, so exact pins matter. Two ready-made setups:

- **Google Colab**: open `TxGNN_Colab.ipynb`. Current Colab runs Python 3.11+, but DGL 0.5.2 and PyTorch 1.8 only have wheels for Python ≤ 3.8, so the notebook installs a 3.8 base via `condacolab` before the CUDA 10.2 stack.
- **macOS, CPU only**: use `requirements-mac-cpu.txt`. These 2020-era wheels are **x86_64 only** (no Apple Silicon builds), so on an M-series Mac you need a Rosetta environment:

```bash
CONDA_SUBDIR=osx-64 conda create -n txgnn_env python=3.8 -y
conda activate txgnn_env
pip install -r requirements-mac-cpu.txt
pip install -e . --no-deps   # --no-deps keeps pandas at 1.3.x, because the code uses DataFrame.append
```

### Quick start: a mini KG for fast testing

The full knowledge graph is 8.1M edges (~945 MB) and a complete run takes hours. To check
that everything works first, `make_mini_kg.py` carves out a smaller subgraph that keeps the
same schema and relation vocabulary, so it exercises the same code paths in about a minute.

```bash
# 1. download the full KG once, then build the subset
mkdir -p data_full
wget -O data_full/kg.csv    https://dataverse.harvard.edu/api/access/datafile/7144484
wget -O data_full/node.csv  https://dataverse.harvard.edu/api/access/datafile/7144482
wget -O data_full/edges.csv https://dataverse.harvard.edu/api/access/datafile/7144483

python make_mini_kg.py --src data_full/kg.csv --out data_mini --n-diseases 400 --seed 42
```

|  | Full KG | Mini KG |
| --- | --- | --- |
| Edges | 8,100,498 | 314,594 |
| Size | 945 MB | 38 MB |
| Relations | 30 | 16 |
| Node types | 5 | 5 (all kept) |

The subset is built so it stays valid for the pipeline: every edge is kept in **both
directions** (`preprocess_kg` de-duplicates by orientation), drug-disease edges are never
downsampled since they are the prediction task, and enough treated diseases are kept for the
5% test split to be non-empty. Tune with `--n-diseases`, `--max-ppi`, `--max-per-rel`.

### Running the pipeline

`test_mini_pipeline.py` runs split → DGL graph → pretrain → finetune → disease-centric
evaluation, saving the best checkpoint (highest validation macro-AUROC) before evaluating.

```bash
# fast smoke test on the mini KG (~1 min on CPU)
python test_mini_pipeline.py --data ./data_mini --device cpu \
    --n-hid 32 --n-inp 32 --n-out 32 --num-walks 10 \
    --pretrain-epochs 1 --finetune-epochs 2 --valid-per-n 1

# full KG with the paper's hyperparameters (needs a GPU, takes hours)
python test_mini_pipeline.py
```

Defaults match `reproduce/train.py`: `n_hid=n_inp=n_out=100`, `proto_num=3`,
`num_walks=200`, 2 pretrain epochs, 500 finetune epochs, `valid_per_n=20`. Every value is a
CLI flag, so `--finetune-epochs 50` gives a shorter run. The device is auto-detected.

### Querying a trained model

`query_txgnn.py` wraps the inference path. TxGNN indexes diseases by `x_idx`, not by name,
so look the index up first:

```bash
python query_txgnn.py --list-diseases asthma
#   idx=2176.0   asthma
#   idx=2117.0   allergic asthma

# train a small model, then rank drugs for a disease
python query_txgnn.py --train --disease "asthma" --relation indication --topk 10
python query_txgnn.py --disease "type 2 diabetes mellitus" --relation contraindication --topk 5
```

The first run trains and saves to `./model_ckpt_mini`, and later runs reload it. `--relation` is
one of `indication`, `contraindication`, `off-label use`. Under the hood this calls:

```python
result = TxEval(model = TxGNN).eval_disease_centric(disease_idxs = [2176.0],
                                                    relation = 'indication',
                                                    save_result = False)
ranked = result['Ranked List'].iloc[0]   # drug names, best first
```

A few things to keep in mind:

- The output format depends on `relation`.
  - `relation='indication'` returns a single DataFrame.
  - `relation=None` returns a dictionary of DataFrames (e.g. `rev_indication`,
    `rev_contraindication`, `rev_off-label use`). The `rev_` prefix indicates a
    disease → drug query, the reverse of the stored drug → disease edges.
- Don't rely on the DataFrame index. Match results using the `Name` column or the row
  order, which follows your input list.
- `eval_disease_centric()` is for evaluation. `AUROC = -1` means the metric could not be
  computed for that disease (e.g. no known drugs in the evaluation split), but the
  `Ranked List` is still generated. For prediction only, use `split='full_graph'`.
- Disease indices are graph-specific, so the same disease has a different index in the
  full and mini KG. IDs stay stable, but looking up by name is the safe habit.

### Tests

`tests/` guards the invariants that are easy to break by accident when editing. The fast
suite builds a small synthetic KG on the fly, so it needs no download and finishes in
seconds.

```bash
python -m unittest discover -s tests -v          # fast suite, ~8 seconds
TXGNN_RUN_SLOW=1 python -m unittest discover -s tests   # adds the real pipeline, ~70 seconds
```

The slow tests need `data_mini/` to exist and are skipped otherwise. No test framework is
required beyond the standard library

What is covered:

- **Graph invariants**: every edge survives in both directions (which `preprocess_kg`
  relies on when it de-duplicates orientations), drug-disease edges are never
  downsampled, the output is a genuine subset, and enough treated diseases remain for a
  non-empty 5% test split.
- **File formats**: `node.csv` is tab separated the way `DataSplitter` reads it, while
  `kg.csv` and `edges.csv` stay comma separated with the full schema.
- **Python 3.8 compatibility**: the source is scanned for 3.9-only string methods such as
  `str.removeprefix`, because DGL 0.5.2 pins the project to 3.8.
- **Requirements**: pins are exact, `setup.py` can parse `requirements.txt`, and the macOS
  file carries torch and dgl plus the wheel index that PyG needs.
- **CLI contracts**: every flag the README documents still exists, and bad input fails
  fast with a readable message instead of a traceback.


### Core API Interface
Using the API, you can (1) reproduce the results in our paper and (2) train TxGNN on your own drug repurposing dataset using a few lines of code, and also generate graph explanations.

```python
from txgnn import TxData, TxGNN, TxEval

# Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = 'complex_disease', seed = 42)
TxGNN = TxGNN(data = TxData,
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = 'cuda:0' # define your cuda device
              )

# Initialize a new model
TxGNN.model_initialize(n_hid = 100, # number of hidden dimensions
                      n_inp = 100, # number of input dimensions
                      n_out = 100, # number of output dimensions
                      proto = True, # whether to use metric learning module
                      proto_num = 3, # number of similar diseases to retrieve for augmentation
                      attention = False, # use attention layer (if use graph XAI, we turn this to false)
                      sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                      agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                      num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                      path_length = 2 # for protein_random_walk sim_measure, define path length
                      )

```

Instead of initializing a new model, you can also load a saved model:

```python
TxGNN.load_pretrained('./model_ckpt')
```

We provide an example pre-trained model weight at [here](https://drive.google.com/file/d/1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj/view).

To do pre-training using link prediction for all edge types, you can type:

```python
TxGNN.pretrain(n_epoch = 2,
               learning_rate = 1e-3,
               batch_size = 1024,
               train_print_per_n = 20)
```

Lastly, to do finetuning on drug-disease relation with metric learning, you can type:

```python
TxGNN.finetune(n_epoch = 500,
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20,
               save_name = finetune_result_path)
```

To save the trained model, you can type:

```python
TxGNN.save_model('./model_ckpt')
```

To evaluate the model on the entire test set using disease-centric evaluation, you can type:

```python
from txgnn import TxEval
TxEval = TxEval(model = TxGNN)
result = TxEval.eval_disease_centric(disease_idxs = 'test_set',
                                     show_plot = False,
                                     verbose = True,
                                     save_result = True,
                                     return_raw = False,
                                     save_name = 'SAVE_PATH')

```

If you want to look at specific disease, you can also do:

```python
result = TxEval.eval_disease_centric(disease_idxs = [9907.0, 12787.0],
                                     relation = 'indication',
                                     save_result = False)
```


After training a satisfying link prediction model, we can also train graph XAI model by:

```python
TxGNN.train_graphmask(relation = 'indication',
                      learning_rate = 3e-4,
                      allowance = 0.005,
                      epochs_per_layer = 3,
                      penalty_scaling = 1,
                      valid_per_n = 20)
```

You can retrieve and save the graph XAI gates (whether or not an edge is important) into a pkl file located as `SAVED_PATH/'graphmask_output_RELATION.pkl'`:

```python
gates = TxGNN.retrieve_save_gates('SAVED_PATH')
```

Of course, you can save and load graphmask model as well via:

```python
TxGNN.save_graphmask_model('./graphmask_model_ckpt')
TxGNN.load_pretrained_graphmask('./graphmask_model_ckpt')

```

### Splits

There are numerous splits prepared in TxGNN. You can switch among them in the `TxData.prepare_split(split = 'XXX', seed = 42)` function.

- `complex_disease` is the systematic split in the paper, where we first sample a set of diseases and then move all of their treatments to test set such that these diseases have zero treatments in training.
- Disease area split first obtains a set of diseases in a disease area using disease ontology and move all of their treatments to the test set and then further removes a fraction of local neighborhood around these diseases to simulate the lack of molecular mechanism characterization of these diseases. There are nine disease areas: `cell_proliferation`, `mental_health`, `cardiovascular`, `anemia`, `adrenal_gland`, `autoimmune`, `metabolic_disorder`, `diabetes`, `neurodigenerative`
- `random` is namely random splits which it randomly shuffles across drug-disease pairs. In the end, most of diseases have seen some treatments in the training set.

During deployment, when evaluate a specific disease, you may want to just mask this disease and use all of the other diseases. In this case, you can use `TxData.prepare_split(split = 'disease_eval', disease_eval_idx = 'XX')` where `disease_eval_idx` is the index of the disease of interest.

Another setting is to train the entire network without any disease masking. You can do that via `split = 'full_graph'`. This will automatically use 95% of data for training and 5% for validation set calculation to do early stopping. No test set is used.


### Cite Us

[MedRxiv preprint](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458)

```
@article{huang2023zeroshot,
  title={Zero-shot Prediction of Therapeutic Use with Geometric Deep Learning and Clinician Centered Design},
  author={Huang, Kexin and Chandak, Payal and Wang, Qianwen and Havaldar, Shreyas and Vaid, Akhil and Leskovec, Jure and Nadkarni, Girish and Glicksberg, Benjamin and Gehlenborg, Nils and Zitnik, Marinka},
  journal = {medRxiv},
  doi = {10.1101/2023.03.19.23287458},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```
