from datasets import load_dataset
import spacy
from collections import Counter
import torch

# Global tokenizers
_spacy_en = None
_spacy_de = None

def _get_spacy_tokenizers():
    """Initialize spacy tokenizers once and cache them."""
    global _spacy_en, _spacy_de
    if _spacy_en is None:
        _spacy_en = spacy.blank('en')
    if _spacy_de is None:
        _spacy_de = spacy.blank('de')
    return _spacy_en, _spacy_de

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
    

