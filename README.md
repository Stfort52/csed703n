# CSED703N - Understanding Large Language Models (Fall 2024)

This repository contains the project materials for the course CSED703N - Understanding Large Language Models (Fall 2024).

## Install

```bash
git clone (repo-url)
cd (repo-name)
pip install -e .
```

## Usage

### Get the data

Clone the [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M) repository to get the data.
You'll likely need git-lfs to clone the repository.
Then, set up a symlink to the data directory.

### launch pretraining

```bash
python -m csed703n.train.pretrain
```

Alternatively, Visual Studio Code users can launch the task `Launch Pretraining` under the command `Tasks: Run Task`.

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
