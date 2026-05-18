"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import bleu
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing) # set gold token to 1 - ε
            true_dist[:, self.pad_idx] = 0  # zero out <pad> token

        return torch.mean(torch.sum(-true_dist * torch.log_softmax(logits, dim=1), dim=1)) # Cross Entropy


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    total_loss = 0
    total_tokens = 0 # for averaging over non-pad tokens
    num_batches = 0

    for batch_idx, (src, tgt) in enumerate(data_iter):
        src, tgt = src.to(device), tgt.to(device)
        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt).to(device)

        with torch.set_grad_enabled(is_train):
            # Slice masks across the last two dimensions (seq_len, seq_len)
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1]) # [batch, tgt_len-1, vocab_size]; teacher forcing: input tgt up to last token
            loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1)) # shift target by 1 for teacher forcing

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        pad_mask = (tgt[:, 1:] != loss_fn.pad_idx)
        total_loss += loss.item() * pad_mask.sum().item() # sum of losses over non-pad tokens
        total_tokens += pad_mask.sum().item()
        num_batches += 1
        
        # Progress indicator every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  [{batch_idx + 1}] loss: {total_loss / total_tokens:.4f}", flush=True)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    # Initialize with <sos>, pre-allocated tensor
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)
    
    for i in range(max_len - 1):  # max_len - 1 excluding <sos>
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model(src, ys, src_mask, tgt_mask)  # [1, seq_len, vocab_size]
        prob = out[:, -1, :]  # [1, vocab_size] — get last time step
        _, next_word = torch.max(prob, dim=1)  # greedy: pick highest prob token
        next_word_item = next_word.item()
        
        # Append next token
        ys = torch.cat([ys, torch.tensor([[next_word_item]], dtype=torch.long, device=device)], dim=1)
        
        if next_word_item == end_symbol:
            break  # stop if <eos> generated
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()
    references = []
    hypotheses = []

    for src, tgt in test_dataloader:
        src, tgt = src.to(device), tgt.to(device)
        src_mask = make_src_mask(src).to(device)
        start_symbol = tgt_vocab.stoi['<sos>']
        end_symbol = tgt_vocab.stoi['<eos>']
        pred_tokens = greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device) # [1, out_len]
        pred_sentence = [tgt_vocab.itos[idx] for idx in pred_tokens.squeeze().tolist() if idx not in (start_symbol, end_symbol)]
        references.append(" ".join([tgt_vocab.itos[idx] for idx in tgt.squeeze().tolist() if idx not in (start_symbol, end_symbol)]))
        hypotheses.append(" ".join(pred_sentence))

    bleu_score = bleu.list_bleu(references, hypotheses)
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    model_config = {
        'src_vocab_size': model.src_vocab_size,
        'tgt_vocab_size': model.tgt_vocab_size,
        'd_model': model.d_model,
        'N': model.N,
        'num_heads': model.num_heads,
        'd_ff': model.d_ff,
        'dropout': model.dropout
    }
    ckpt_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model_config
    }
    torch.save(ckpt_dict, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    ckpt_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_dict['model_state_dict'])
    if optimizer is not None and ckpt_dict['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    if scheduler is not None and ckpt_dict['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
    return ckpt_dict['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(quick_test: bool = False) -> None:
    """
    Set up and run the full training experiment.

    Args:
        quick_test: If True, use only first 100 samples from each split for quick iteration. (Optional)

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    import time
    
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Quick test mode: {quick_test}")
    
    start_time = time.time()
    
    # Dataset and vocab
    max_samples = 100 if quick_test else None
    print("\n=== Loading Datasets ===")
    train_dataset = Multi30kDataset('train', max_samples=max_samples)
    val_dataset = Multi30kDataset('validation', max_samples=max_samples)
    test_dataset = Multi30kDataset('test', max_samples=max_samples)

    # Build vocab only on the training split and reuse for val/test
    print("\n=== Building Vocabulary ===")
    train_dataset.build_vocab()
    # Copy vocab mappings to ensure consistent token indices across splits
    for ds in (val_dataset, test_dataset):
        ds.de_stoi = train_dataset.de_stoi
        ds.en_stoi = train_dataset.en_stoi
        ds.de_itos = train_dataset.de_itos
        ds.en_itos = train_dataset.en_itos
        ds.special_tokens = train_dataset.special_tokens

    config = {
        'src_vocab_size': len(train_dataset.de_stoi),
        'tgt_vocab_size': len(train_dataset.en_stoi),
        'd_model': 128,
        'N': 1,
        'num_heads': 1,
        'd_ff': 256,
        'dropout': 0.1,
        'num_epochs': 1,
        'batch_size': 1,
        'learning_rate': 0.9, # base LR for NoamScheduler
        'quick_test': quick_test,
    }
    wandb.init(project="da6401-a3", config=config)

    # Get processed datasets
    print("\n=== Processing Data ===")
    train_dataset.process_data()
    val_dataset.process_data()
    test_dataset.process_data()
    train_df = train_dataset.processed_data
    val_df = val_dataset.processed_data
    test_df = test_dataset.processed_data

    # DataLoaders
    print("\n=== Creating DataLoaders ===")
    train_loader = DataLoader(train_df, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_df, batch_size=config['batch_size'], shuffle=False, collate_fn=val_dataset.collate_fn)
    test_loader = DataLoader(test_df, batch_size=config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # Model, optimizer, scheduler, loss
    print("\n=== Initializing Model ===")
    model = Transformer(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=config['d_model'], warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(vocab_size=config['tgt_vocab_size'], pad_idx=1, smoothing=0.1) # <pad> index is 1

    prep_time = time.time() - start_time
    print(f"\nPreparation time: {prep_time:.2f}s")
    print(f"\n=== Starting Training ===")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s")
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)
        save_checkpoint(model, optimizer, scheduler, epoch)
    
    # Final BLEU evaluation on test set
    print(f"\n=== Final BLEU Evaluation ===")
    tgt_vocab = type('Vocab', (), {'stoi': test_dataset.en_stoi, 'itos': {idx: token for token, idx in test_dataset.en_stoi.items()}})
    bleu = evaluate_bleu(model, test_loader, tgt_vocab=tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    wandb.finish()

if __name__ == "__main__":
    import sys
    quick_test = "--quick-test" in sys.argv or "-q" in sys.argv
    run_training_experiment(quick_test=quick_test)
