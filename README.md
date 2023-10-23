<div align="center">

# Challenges in Context-Aware Neural Machine Translation
</div>
<p align="center">
<a href="LICENSE" alt="MIT License"><img src="https://img.shields.io/badge/license-MIT-FAD689.svg" /></a>
<a href="https://arxiv.org/abs/2305.13751" alt="Paper Link"><img src="https://img.shields.io/badge/paper-link-D9AB42" /></a>
</p>

Authors: [Linghao Jin](), [Jacqueline He](https://jacqueline-he.github.io/), [Jonathan May](https://www.isi.edu/directory/jonmay/), [Xuezhe Ma](https://xuezhemax.github.io/)

This repository contains the code for our EMNLP 2023 paper, ["Challenges in Context-Aware Neural Machine Translation"](https://arxiv.org/pdf/2210.14975.pdf). 

Context-aware neural machine translation, a paradigm that involves leveraging information beyond sentence-level context to resolve inter-sentential discourse dependencies and improve document-level translation quality, has given rise to a number of recent techniques. 
However, despite well-reasoned intuitions, most context-aware translation models yield only modest improvements over sentence-level systems. 
In this work, we investigate and present several core challenges, relating to discourse phenomena, context usage, model architectures, and document-level evaluation, that impede progress within the field. 
To address these problems, we propose a more realistic setting for document-level translation, called paragraph-to-paragraph (Para2Para) translation, and collect a new dataset of Chinese-English novels to promote future research.


## Table of Contents
  * [Quick Start](#quick-start)
  * [Context-aware NMT](#context-aware-nmt)
	+ [Data](#data)
    + [Training](#training)
    + [Evaluation](#evaluation)
  * [P2P NMT](#p2p-nmt)
    + [P2P Data](#p2p-data)
    + [Pre-training](#pre-training)
    + [Fine-tuning](#fine-tuning)
    + [P2P Evaluation](#p2p-evaluation)
  * [Code Acknowledgements](#code-acknowledgements)
  * [Citation](#citation)

## Quick Start
```bash
conda create -n canmt python=3.8
conda activate canmt
pip install -r requirements.txt
```

Note: We use `fairseq 0.9.0`, so as to be compatible with the [Mega](https://arxiv.org/abs/2209.10655) (Ma et al., 2022) architecture. To download the official version:

```python
git clone https://github.com/facebookresearch/mega.git && cd mega
pip install --editable ./
```

## Context-aware NMT

### Data
We provide sentence counts for the train/valid/test splits on the datasets used in this paper below:

| Dataset | Lg. Pair | Train    | Valid | Test |
|---------|----------|----------|-------|------|
| BWB [(Jiang et al., 2022)](https://arxiv.org/abs/2210.14667)     | Zh->En   | 9576566  | 2632  | 2618 |
| WMT17 [(Bojar et al., 2017)](https://aclanthology.org/W17-4717/) | Zh->En   | 25134743 | 2002  | 2001 |
| IWSLT17 [(Cettolo et al., 2012)](https://aclanthology.org/2017.iwslt-1.1/) | En<->Fr  | 232825   | 5819  | 1210 |
| IWSLT17 [(Cettolo et al., 2012)](https://aclanthology.org/2017.iwslt-1.1/) | En<->De  | 206112   | 5431  | 1080 |

Please refer to `canmt-data/README.md` for details on downloading and processing the datasets.

### Training

**Run training script**
Train all models for lg. pair *zh->en* implemented in the paper, you can run the following script:
```bash
cd sh/zh-en
chmod +x train_all.sh 
./train_all.sh
```
You can configure the hyper-parameters in `train_all.sh` accordingly. Models are saved to `ckpt/`. 

You can also train each model and setting separatively using the following scripts! 
Note: N, M are source and target context sizes, respectively. Following Fernandes et al., 2021, our settings are 0-1 (denoted as 1-2 in the paper) , and 1-1 (denoted as 2-2 in the paper).

**XFMR-based context-aware *concatenation* baseline** (`concat_models`).
```bash
cd sh/zh-en
chmod +x train_concat.sh
./train_concat.sh
```
**Mega-based context-aware *concatenation* baseline** (`concat_models`)
```bash
cd sh/zh-en
chmod +x train_mega.sh
./train_mega.sh
```

### Evaluation
To evaluate all trained models with the lg. pair *zh->en* on BLEU, COMET and BlonDe, you can run the following script
```bash
cd sh/zh-en
chmod +x generate_all.sh 
./generate_all.sh
```


## P2P NMT

### P2P Data
| Title | Pub. Year | Pub. Year | Avg. Para. Length |
|------------------------------------|-----|----|--------------|
| *Gone with the Wind* (Margaret Mitchell) | 1936 | 3556 | 143 |
| *Rebecca* (Daphne du Maurier) | 1938 | 1237 | 157 |
| *Alice’s Adventure in Wonderland* (Lewis Carroll) | 1865 | 218 | 144 |
| *Foundation* (Isaac Asimov) | 1951 | 3413 | 76 |
| *A Tale of Two Cities* (Charles Dickens) | 1859 | 696 | 225 |
| *Twenty Thousand Leagues Under the Seas* (Jules Verne) | 1870 | 1425 | 117 |

### Pre-training

We use the following backbone architectures for pre-training before fine-tuning on Para2Para dataset:

- **XFMR** (Vaswani et al., 2017), the Transformer-BIG model
- **LIGHTCONV**  (Wu et al., 2019), which replaces the self-attention modules in the Transformer-BIG with fixed convolutions
- **MBART25** (Liu et al., 2020), which is pre-trained on 25 languages at the document level

### Fine-tuning
```bash
cd sh/p2p
chmod +x train_all.sh
./train_all.sh
```
### P2P Evaluation
We provide the scripts to evaluate the pre-trained models on Para2Para without fine-tuning:
```bash
cd sh/p2p
chmod +x generate_pretrained.sh
./generate_pretrained.sh
```

To evaluate the fine-tuned models on Para2Para:
```bash
cd sh/p2p
chmod +x generate_finetuned.sh
./generate_finetuned.sh
```

**Collective results:**
![p2p results](p2p_results.png)

## Code Acknowledgements
- [contextual_mt](https://github.com/neulab/contextual-mt) package from Fernandes et al., 2021
- [BlonDe](https://github.com/EleanorJiang/BlonDe) package from Jiang et al., 2022
- [MEGA](https://github.com/facebookresearch/mega) package from Ma et al., 2023

## Citation
```bibtex
@inproceedings{jin2023challenges,
   title={Challenges in Context-Aware Neural Machine Translation},
   author={Jin, Linghao and He, Jacqueline and May, Jonathan and Ma, Xuezhe},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2023}
}
```
