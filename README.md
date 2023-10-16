
## Challenges in Context-Aware Neural Machine Translation

Authors: [Linghao Jin](), [Jacqueline He](https://jacqueline-he.github.io/), [Jonathan May](https://www.isi.edu/~jonmay/), [Xuezhe Ma](https://xuezhemax.github.io/)

This repository contains the code for our EMNLP 2023 paper, ["Challenges in Context-Aware Neural Machine Translation"](https://arxiv.org/pdf/2210.14975.pdf). 

Context-aware neural machine translation, a paradigm that involves leveraging information beyond sentence-level context to resolve inter-sentential discourse dependencies and improve document-level translation quality, has given rise to a number of recent techniques. 
However, despite well-reasoned intuitions, most context-aware translation models yield only modest improvements over sentence-level systems. 
In this work, we investigate and present several core challenges, relating to discourse phenomena, context usage, model architectures, and document-level evaluation, that impede progress within the field. 
To address these problems, we propose a more realistic setting for document-level translation, called paragraph-to-paragraph (Para2Para) translation, and collect a new dataset of Chinese-English novels to promote future research.


## Table of Contents
  * [Quick Start](#quick-start)
  * [Context-aware NMT](#context-aware)
	  + [Data](#context-data)
    + [Training](#context-training)
    + [Evaluation](#context-evaluation)
  * [P2P NMT](#p2p)
    + [Data](#p2p-data)
    + [Pre-training](#p2p-pretraining)
    + [Fine-tuning](#p2p-finetuning)
    + [Evaluation](#p2p-evaluation)
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
| Dataset | Lg. pair | Train    | Valid | Test |
|---------|----------|----------|-------|------|
| BWB     | Zh->En   | 9576566  | 2632  | 2618 |
| WMT17   | Zh->En   | 25134743 | 2002  | 2001 |
| IWSLT17 | En<->Fr  | 232825   | 5819  | 1210 |
| IWSLT17 | En<->De  | 206112   | 5431  | 1080 |

### Training


**1. Run training script**
Train all models for lg. pair *zh->en* implemented in the paper, you can run the following script
```bash
cd sh/zh-en
chmod +x train_all.sh 
./train_all.sh
```
You can configure the hyper-parameters in `train_all.sh` accordingly. Models are saved to `ckpt/`. 

You can also train each model and setting separatively using the following scripts! 
Note: N, M are source and target context sizes, respectively. Following Fernandes et al., our settings are 0-1 (representing the 1-2 in the paper) , and 1-1 (representing the 2-2 in the paper).

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


## P2P NMT

### Data
| Title | Pub. Year | Pub. Year | Avg. Para. Length |
|------------------------------------|-----|----|--------------|
| *Gone with the Wind* (Margaret Mitchell) | 1936 | 3556 | 143 |
| *Rebecca* (Daphne du Maurier) | 1938 | 1237 | 157 |
| *Alice’s Adventure in Wonderland* (Lewis Carroll) | 1865 | 218 | 144 |
| *Foundation* (Isaac Asimov) | 1951 | 3413 | 76 |
| *A Tale of Two Cities* (Charles Dickens) | 1859 | 696 | 225 |
| *Twenty Thousand Leagues Under the Seas* (Jules Verne) | 1870 | 1425 | 117 |

### Pre-training

### Fine-tuning

### Evaluation

## Code Acknowledgements


## Citation
```bibtex
@inproceedings{jin2023challenges,
   title={Challenges in Context-Aware Neural Machine Translation},
   author={Jin, Linghao and He, Jacqueline and May, Jonathan and Ma, Xuezhe},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2023}
}
```
