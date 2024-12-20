# CSED703N - Understanding Large Language Models (Fall 2024)

This repository contains the project materials for the course CSED703N - Understanding Large Language Models (Fall 2024).

## Install

```bash
git clone https://github.com/Stfort52/csed703n
cd csed703n
pip install -e .
```

It's highly recommended to use a virtual environment. To also install the dev dependencies, run `pip install -e .[dev]` instead.

## Usage

### Get the data

Clone the [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M) repository to get the data.
You'll likely need git-lfs to clone the repository.
Then, set up a symlink to the required files in the data directory like below:
You should be able to easily locate the required files in the GeneCorpus-30M repository.

```bash
data
├── datasets
│   ├── genecorpus_30M_2048.dataset -> /path/to/30M/dataset
│   ├── iCM_diff_dropseq.dataset -> /path/to/dropseq/dataset
│   └── panglao_SRA553822-SRS2119548.dataset -> /path/to/panglao/dataset
├── is_bivalent.csv
└── token_dictionary.pkl -> /path/to/token/dictionary
```

#### Optional: subset the data

The full GeneCorpus-30M dataset is quite large. Therefore, the one-thirtieth subset of the dataset is used in the project. You can subset it by running the notebook at `notebooks/subset_genecorpus.ipynb`.

### launch pretraining

```bash
python -m csed703n.train.pretrain
```

Alternatively, Visual Studio Code users can launch the task `Launch Pretraining` under the command `Tasks: Run Task`.

This will create new version of the model and save it to the `checkpoints` directory.

#### pretrain with DDP

To launch pretraining with DDP, run the following command:

```bash
bash -c csed703n/train/ddp.sh <master_port> <hosts> pretrain
```

Alternatively, Visual Studio Code users can launch the task `Distributed Pretraining` under the command `Tasks: Run Task`.

### launch finetuning

```bash
python -m csed703n.train.finetune
```

Alternatively, Visual Studio Code users can launch the task `Launch Fine-tuning` under the command `Tasks: Run Task`.

#### fine-tune with DDP

To launch finetuning with DDP, run the following command:

```bash
bash -c csed703n/train/ddp.sh <master_port> <hosts> finetune
```

Alternatively, Visual Studio Code users can launch the task `Distributed Fine-tuning` under the command `Tasks: Run Task`.

## Advanced Usage

### Configure the model

The base model has the following configurations, respecting the original paper "Transfer learning enables predictions in network biology" (<https://doi.org/10.1038/s41586-023-06139-9>)

```yaml
config:
  absolute_pe_kwargs:
    embed_size: 256
    max_len: 2048
  absolute_pe_strategy: trained
  act_fn: relu
  attn_dropout: 0.02
  d_ff: 512
  d_model: 256
  ff_dropout: 0.02
  n_vocab: 25426
  norm: post
  num_heads: 4
  num_layers: 6
  relative_pe_kwargs: {}
  relative_pe_shared: true
  relative_pe_strategy: null
  tupe: false
ignore_index: -100
initialization_range: 0.02
lr: 0.001
lr_scheduler: linear
warmup_steps_or_ratio: 0.1
weight_decay: 0.001
```

The `config` key contains the model configuration. Anything else is a hyperparameter used for training. You can edit the configuration by editing the pretraining script (`csed703n/train/pretrain.py`) or the finetuning script (`csed703n/train/finetune.py`).

#### PE Configuration

6 parameters control the positional encoding (PE) strategy:

- `absolute_pe_strategy` and `absolute_pe_kwargs` for the absolute PE.
  - valid values: `None`, `"trained"`, `"sinusoidal"`.

- `relative_pe_strategy` and `relative_pe_kwargs` for the relative PE.
  - valid values: `None`, `"trained"`, `"sinusoidal"`, `"t5"`.
  
- `relative_pe_shared` is a Bool for whether to share the relative PE weights across layers.
- `tupe`: Bool for whether to apply the TUPE method from the paper "Rethinking Positional Encoding in Language Pre-training" (<https://arxiv.org/abs/2006.15595>)
  - This requires an absolute PE to be set
  - Without a relative PE, this will behave like the `TUPE-A` model
  - With a relative PE, this will behave like the `TUPE-R` model
