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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import bleu
import wandb
import matplotlib.pyplot as plt

from model import Transformer, MultiHeadAttention, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler


# Helper functions for logging and visualization

def _make_tgt_vocab(dataset) -> object:
    return type(
        'Vocab',
        (),
        {'stoi': dataset.en_stoi, 'itos': {idx: token for token, idx in dataset.en_stoi.items()}},
    )()

def _log_qk_grad_norms(model: Transformer, step: int) -> None:
    """Mean L2 norm of W_Q and W_K gradients across all attention modules."""
    q_norms, k_norms = [], []
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            if module.W_Q.weight.grad is not None:
                q_norms.append(module.W_Q.weight.grad.norm().item())
            if module.W_K.weight.grad is not None:
                k_norms.append(module.W_K.weight.grad.norm().item())
    if q_norms:
        wandb.log(
            {
                'grad_norm/W_Q': sum(q_norms) / len(q_norms),
                'grad_norm/W_K': sum(k_norms) / len(k_norms),
            },
            step=step,
        )

def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]['lr']

def _last_encoder_self_attn(model: Transformer) -> MultiHeadAttention:
    return model.encoder.layers[-1].self_attn

def _log_attention_heatmaps(model: Transformer, step: int, sample_idx: int = 0) -> None:
    """Log per-head heatmaps from the last encoder layer self-attention only."""
    module = _last_encoder_self_attn(model)
    if not hasattr(module, 'last_attn_weights'):
        return

    logs = {}
    # [batch, heads, seq_q, seq_k] — one example from the batch
    weights = module.last_attn_weights[sample_idx].cpu()
    layer_idx = len(model.encoder.layers) - 1
    for head_idx in range(weights.size(0)):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(weights[head_idx].numpy(), aspect='auto', cmap='viridis')
        ax.set_title(f'encoder layer {layer_idx} — head {head_idx}')
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        plt.colorbar(im, ax=ax)
        logs[f'attention_heatmap/encoder_last/head_{head_idx}'] = wandb.Image(fig)
        plt.close(fig)
    if logs:
        wandb.log(logs, step=step)

@torch.no_grad()
def _capture_attention_maps(
    model: Transformer,
    src: torch.Tensor,
    tgt_in: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> None:
    """Forward pass storing attention weights from last encoder self-attn only."""
    last_attn = _last_encoder_self_attn(model)
    last_attn.store_attn_weights = True
    model.eval()
    model(src, tgt_in, src_mask, tgt_mask)
    last_attn.store_attn_weights = False

def _log_correct_token_prob(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
    step: int,
) -> float:
    """
    Mean softmax probability assigned to the gold token (non-pad positions).
    logits: [batch, seq_len, vocab] or [batch*seq, vocab]
    target: [batch, seq_len] or [batch*seq]
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
    probs = F.softmax(logits, dim=-1)
    correct_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = target != pad_idx
    if mask.sum() == 0:
        return 0.0
    mean_prob = correct_prob[mask].mean().item()
    wandb.log({'correct_token_prob': mean_prob}, step=step)
    return mean_prob

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
    global_step: int = 0,
    log_batch_metrics: bool = False,
    log_lr: bool = False,
    log_qk_grads: bool = False,
    max_grad_log_steps: int = 1000,
    log_correct_token_prob: bool = False,
) -> tuple[float, int]:
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
        (avg_loss, global_step) : Average loss over the epoch and updated global step.

    """
    model.train() if is_train else model.eval()
    total_loss = 0
    total_tokens = 0 # for averaging over non-pad tokens
    num_batches = 0

    for batch_idx, (src, tgt) in enumerate(data_iter):
        src, tgt = src.to(device), tgt.to(device)
        src_mask = make_src_mask(src)
        tgt_in = tgt[:, :-1]
        tgt_mask = make_tgt_mask(tgt_in)

        with torch.set_grad_enabled(is_train):
            output = model(src, tgt_in, src_mask, tgt_mask) # [batch, tgt_len-1, vocab_size]; teacher forcing
            loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1)) # shift target by 1 for teacher forcing

            if is_train:
                if log_correct_token_prob:
                    _log_correct_token_prob(
                        output, tgt[:, 1:], loss_fn.pad_idx, global_step
                    )
                optimizer.zero_grad()
                loss.backward()
                if log_qk_grads and global_step < max_grad_log_steps:
                    _log_qk_grad_norms(model, global_step)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if log_lr:
                    wandb.log({'learning_rate': _current_lr(optimizer)}, step=global_step - 1)

        pad_mask = (tgt[:, 1:] != loss_fn.pad_idx)
        n_tokens = pad_mask.sum().item()
        batch_loss = loss.item()
        total_loss += batch_loss * n_tokens
        total_tokens += n_tokens
        num_batches += 1

        if log_batch_metrics:
            prefix = 'train' if is_train else 'val'
            step = (global_step - 1) if is_train else global_step + batch_idx
            wandb.log({f'{prefix}_loss/batch': batch_loss}, step=step)
        
        # Progress indicator every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  [{batch_idx + 1}] loss: {total_loss / total_tokens:.4f}", flush=True)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, global_step


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
        tgt_mask = make_tgt_mask(ys)
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
        src_mask = make_src_mask(src)
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
    val_bleu: Optional[float] = None,
    train_config: Optional[dict] = None,
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
        'scheduler_state_dict', 'model_config', 'val_bleu', 'train_config'

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
        'dropout': model.dropout,
        'scale_attention': model.scale_attention,
        'positional_encoding_type': model.positional_encoding_type,
    }
    ckpt_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model_config,
        'val_bleu': val_bleu,
        'train_config': train_config,
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

_DEFAULT_EXP = {
    'scheduler': 'noam',
    'scale_attention': True,
    'positional_encoding': 'sinusoidal',
    'label_smoothing': 0.1,
    'log_lr': False,
    'log_qk_grads': False,
    'log_attn_heatmaps': False,
    'log_correct_token_prob': False,
}

EXPERIMENT_CONFIGS = {
    'exp1': {},
    'exp21a': {'log_lr': True},
    'exp21b': {'scheduler': 'fixed', 'log_lr': True},
    'exp22a': {'log_qk_grads': True},
    'exp22b': {'scale_attention': False, 'log_qk_grads': True},
    'exp23': {'log_attn_heatmaps': True},
    'exp24a': {'positional_encoding': 'sinusoidal'},
    'exp24b': {'positional_encoding': 'learned'},
    'exp25a': {'label_smoothing': 0.1, 'log_correct_token_prob': True},
    'exp25b': {'label_smoothing': 0.0, 'log_correct_token_prob': True},
}


def _resolve_experiment_config(experiment: str) -> dict:
    if experiment not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {list(EXPERIMENT_CONFIGS)}")
    cfg = _DEFAULT_EXP.copy()
    cfg.update(EXPERIMENT_CONFIGS[experiment])
    return cfg


def run_training_experiment(experiment: str = 'exp1', quick_test: bool = False) -> None:
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

    exp_cfg = _resolve_experiment_config(experiment)

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment: {experiment}")
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

    use_noam = exp_cfg['scheduler'] == 'noam'
    config = {
        'experiment': experiment,
        'src_vocab_size': len(train_dataset.de_stoi),
        'tgt_vocab_size': len(train_dataset.en_stoi),
        'd_model': 512,
        'N': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'num_epochs': 30,
        'batch_size': 64,
        'learning_rate': 1.0 if use_noam else 1e-4,
        'scheduler': exp_cfg['scheduler'],
        'scale_attention': exp_cfg['scale_attention'],
        'positional_encoding': exp_cfg['positional_encoding'],
        'label_smoothing': exp_cfg['label_smoothing'],
        'warmup_steps': 4000 if use_noam else 0,
        'early_stopping_patience': 5,
        'quick_test': quick_test,
    }
    wandb.init(project="da6401-a3", name=experiment, config=config)

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
        dropout=config['dropout'],
        scale_attention=config['scale_attention'],
        positional_encoding_type=config['positional_encoding'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98), eps=1e-9)
    scheduler = None
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=config['d_model'], warmup_steps=config['warmup_steps'])
    loss_fn = LabelSmoothingLoss(
        vocab_size=config['tgt_vocab_size'],
        pad_idx=1,
        smoothing=config['label_smoothing'],
    )
    tgt_vocab = _make_tgt_vocab(val_dataset)
    log_batch_metrics = True  # all experiments log loss/BLEU curves
    log_lr = exp_cfg['log_lr']
    log_qk_grads = exp_cfg['log_qk_grads']
    log_attn_heatmaps = exp_cfg['log_attn_heatmaps']
    log_correct_token_prob = exp_cfg['log_correct_token_prob']
    # Fixed batch for attention visualization (first val sample)
    attn_vis_batch = None
    if log_attn_heatmaps:
        src_vis, tgt_vis = next(iter(val_loader))
        attn_vis_batch = (
            src_vis[:1].to(device),
            tgt_vis[:1].to(device),
        )

    prep_time = time.time() - start_time
    print(f"\nPreparation time: {prep_time:.2f}s")
    print(f"\n=== Starting Training ===")
    
    # Training loop
    global_step = 0
    best_bleu = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = f"best_checkpoint_{experiment}.pt"

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, global_step = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, epoch,
            is_train=True, device=device, global_step=global_step,
            log_batch_metrics=log_batch_metrics, log_lr=log_lr, log_qk_grads=log_qk_grads,
            log_correct_token_prob=log_correct_token_prob,
        )
        val_loss, _ = run_epoch(
            val_loader, model, loss_fn, None, None, epoch,
            is_train=False, device=device, global_step=global_step,
            log_batch_metrics=log_batch_metrics,
        )
        val_bleu = evaluate_bleu(model, val_loader, tgt_vocab=tgt_vocab, device=device)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.4f} "
            f"- Val Loss: {val_loss:.4f} - Val BLEU: {val_bleu:.2f} - Time: {epoch_time:.2f}s"
        )
        wandb.log(
            {
                'train_loss/epoch': train_loss,
                'val_loss/epoch': val_loss,
                'bleu/epoch': val_bleu,
            },
            step=global_step,
        )
        if log_attn_heatmaps and attn_vis_batch is not None:
            src_vis, tgt_vis = attn_vis_batch
            tgt_in_vis = tgt_vis[:, :-1]
            _capture_attention_maps(
                model,
                src_vis,
                tgt_in_vis,
                make_src_mask(src_vis),
                make_tgt_mask(tgt_in_vis),
            )
            _log_attention_heatmaps(model, step=global_step)
            model.train()

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                path=best_checkpoint_path,
                val_bleu=val_bleu,
                train_config=config,
            )
            print(f"  New best val BLEU: {best_bleu:.2f} → saved {best_checkpoint_path}")
            wandb.log({'best_val_bleu': best_bleu}, step=global_step)
        else:
            epochs_without_improvement += 1
            print(
                f"  No BLEU improvement ({epochs_without_improvement}/"
                f"{config['early_stopping_patience']})"
            )

        if epochs_without_improvement >= config['early_stopping_patience']:
            print(
                f"\nEarly stopping: no val BLEU improvement for "
                f"{config['early_stopping_patience']} epochs (best={best_bleu:.2f})"
            )
            break

    # Final BLEU evaluation on test set (best checkpoint)
    print(f"\n=== Final BLEU Evaluation (best checkpoint) ===")
    if best_bleu >= 0:
        load_checkpoint(best_checkpoint_path, model, optimizer, scheduler)
        print(f"Loaded best checkpoint (val BLEU={best_bleu:.2f})")
    test_vocab = _make_tgt_vocab(test_dataset)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab=test_vocab, device=device)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({'test_bleu': test_bleu, 'best_val_bleu': best_bleu}, step=global_step)
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    wandb.finish()

if __name__ == "__main__":
    import argparse

    # exp1: full training of base
    # exp21a: Use Noam Scheduler
    # exp21b: Use fixed lr of 10^-4 with no warmup
    # exp22a: Use scaling factor in attention
    # exp22b: Without scaling factor in attention
    # exp23: Visualize each head in NHA
    # exp24a: Use sinusoidal positional encoding
    # exp24b: Use learned positional encoding
    # exp25a: Train with Label smoothing e=0.1
    # exp25b: Train with Label smoothing e=0.0
    parser = argparse.ArgumentParser(description="Train Transformer for DA6401 A3")
    parser.add_argument(
        '--experiment',
        choices=list(EXPERIMENT_CONFIGS.keys()),
        default='exp1',
        help='Which experiment configuration to run',
    )
    parser.add_argument('--quick-test', action='store_true', help='Use 100 samples per split')
    args = parser.parse_args()
    run_training_experiment(experiment=args.experiment, quick_test=args.quick_test)
