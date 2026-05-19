import os

from datasets import load_dataset
import spacy
from collections import Counter
import torch

# Pre-built Multi30k train vocab (~0.03s load); required for fast infer()
VOCAB_CACHE_PATH = os.path.join(os.path.dirname(__file__), "vocab", "multi30k_vocab.pt")

# Global tokenizers
_spacy_en = None
_spacy_de = None
_infer_vocab = None

def _get_spacy_tokenizers():
    """Initialize spacy tokenizers once and cache them."""
    global _spacy_en, _spacy_de
    if _spacy_en is None:
        _spacy_en = spacy.blank('en')
    if _spacy_de is None:
        _spacy_de = spacy.blank('de')
    return _spacy_en, _spacy_de


def get_infer_vocab():
    """
    Return vocab + spacy tokenizers for inference without loading HF data
    or rebuilding vocab from 29k sentences (autograder 3s limit).
    """
    global _infer_vocab
    if _infer_vocab is not None:
        return _infer_vocab

    vocab = Multi30kDataset.__new__(Multi30kDataset)
    vocab.spacy_en, vocab.spacy_de = _get_spacy_tokenizers()

    if not os.path.isfile(VOCAB_CACHE_PATH):
        raise FileNotFoundError(
            f"Missing {VOCAB_CACHE_PATH}. Run once: "
            "python -c \"from dataset import build_vocab_cache; build_vocab_cache()\""
        )

    try:
        cache = torch.load(VOCAB_CACHE_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        cache = torch.load(VOCAB_CACHE_PATH, map_location="cpu")
    vocab.de_stoi = cache["de_stoi"]
    vocab.en_stoi = cache["en_stoi"]
    vocab.de_itos = cache["de_itos"]
    vocab.en_itos = cache["en_itos"]
    vocab.special_tokens = cache["special_tokens"]
    _infer_vocab = vocab
    return _infer_vocab


def build_vocab_cache(path: str = VOCAB_CACHE_PATH) -> str:
    """Build and save Multi30k train vocabulary (for local regeneration only)."""
    ds = Multi30kDataset("train")
    ds.build_vocab()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "de_stoi": ds.de_stoi,
            "en_stoi": ds.en_stoi,
            "de_itos": ds.de_itos,
            "en_itos": ds.en_itos,
            "special_tokens": ds.special_tokens,
        },
        path,
    )
    return path


class Multi30kDataset:
    def __init__(self, split='train', max_samples=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        
        Args:
            split: 'train', 'validation', or 'test'
            max_samples: If set, only use first N samples (optional)
        """
        self.split = split
        self.max_samples = max_samples
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        print(f"Loading {split} dataset...", end=" ", flush=True)
        self.dataset = load_dataset('bentrevett/multi30k', split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        print(f"({len(self.dataset)} samples)")
        # Get cached tokenizers
        self.spacy_en, self.spacy_de = _get_spacy_tokenizers()

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        print(f"Building vocabulary for {self.split}...", end=" ", flush=True)
        de_counter = Counter()
        en_counter = Counter()
        for i, example in enumerate(self.dataset):
            if (i + 1) % 5000 == 0:
                print(f"{i+1}...", end=" ", flush=True)
            de_tokens = [token.text.lower() for token in self.spacy_de(example['de'])]
            en_tokens = [token.text.lower() for token in self.spacy_en(example['en'])]
            de_counter.update(de_tokens)
            en_counter.update(en_tokens)

        # Build vocab dicts
        self.de_stoi = {}
        self.en_stoi = {}
        # Add special tokens
        self.special_tokens = {
            '<unk>': 0,
            '<pad>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        for token, idx in self.special_tokens.items():
            self.de_stoi[token] = idx
            self.en_stoi[token] = idx
        # Add regular tokens
        idx = len(self.special_tokens)
        for word, _ in de_counter.items():
            if word not in self.de_stoi:
                self.de_stoi[word] = idx
                idx += 1

        idx = len(self.special_tokens)
        for word, _ in en_counter.items(): # Can use freq if we want to limit vocab size based on frequency of occurrence
            if word not in self.en_stoi:
                self.en_stoi[word] = idx
                idx += 1
        print(f"Done! DE vocab: {len(self.de_stoi)}, EN vocab: {len(self.en_stoi)}")

        self.de_itos = {idx: token for token, idx in self.de_stoi.items()}
        self.en_itos = {idx: token for token, idx in self.en_stoi.items()}


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        print(f"Processing {self.split} data...", end=" ", flush=True)
        self.processed_data = []
        for i, example in enumerate(self.dataset):
            if (i + 1) % 5000 == 0:
                print(f"{i+1}...", end=" ", flush=True)
            # Tokenize
            de_tokens = [token.text.lower() for token in self.spacy_de(example['de'])]
            en_tokens = [token.text.lower() for token in self.spacy_en(example['en'])]
            # Add SOS/EOS
            de_tokens = ['<sos>'] + de_tokens + ['<eos>']
            en_tokens = ['<sos>'] + en_tokens + ['<eos>']
            # Convert to indices
            de_indices = [self.de_stoi.get(token, self.special_tokens['<unk>']) for token in de_tokens]
            en_indices = [self.en_stoi.get(token, self.special_tokens['<unk>']) for token in en_tokens]

            self.processed_data.append({"src": de_indices, "tgt": en_indices})
        print("Done!")

    def _pad_sequences(self, sequences, pad_token):
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
        return padded_seqs
      
    def collate_fn(self, batch):
        """
        Collate function to pad sequences in a batch to the same length.
        """
        src_batch = [item['src'] for item in batch]
        tgt_batch = [item['tgt'] for item in batch]
        # Pad sequences
        src_padded = self._pad_sequences(src_batch, self.special_tokens['<pad>'])
        tgt_padded = self._pad_sequences(tgt_batch, self.special_tokens['<pad>'])
        return torch.tensor(src_padded), torch.tensor(tgt_padded)
    

