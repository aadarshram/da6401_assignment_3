# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

Github Repo: https://github.com/aadarshram/da6401_assignment_3 
Wandb Report: https://wandb.ai/ramachandranaadarsh-indian-institute-of-technology-madras/da6401-a3/reports/Assignment-3--VmlldzoxNjkzNDE2NQ 

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── vocab/
│   └── multi30k_vocab.pt  # Pre-built train vocab (required for fast infer())
├── train.py           # Training loops and Greedy Decoding inference
```

Regenerate the vocab cache after changing tokenization:

```bash
python -c "from dataset import build_vocab_cache; build_vocab_cache()"
```

### Autograder submission (BLEU ≥ 20)

The autograder calls `model.infer(german_sentence)` on a **trained** model. BLEU ≈ 0 means weights were **not loaded** (random init).

Include in your zip:

| File | Purpose |
|------|---------|
| `checkpoint.pt` | Trained weights (`save_checkpoint` format) |
| `vocab/multi30k_vocab.pt` | Fast infer tokenization |

After training, copy your best file to the name the autograder expects:

```bash
cp best_checkpoint_exp1.pt checkpoint.pt
```

`Transformer()` auto-loads `checkpoint.pt` next to `model.py` if present; the autograder may also call `load_checkpoint("checkpoint.pt", model)`.
```
