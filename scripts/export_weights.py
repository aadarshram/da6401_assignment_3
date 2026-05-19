#!/usr/bin/env python3
"""Export model weights only from a full training checkpoint (drops optimizer/scheduler)."""

import argparse
import os
import torch


def export_weights(src: str, dst: str, fp16: bool = False) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if fp16:
        state = {k: v.half() for k, v in state.items()}
    out = {
        "model_state_dict": state,
        "model_config": ckpt.get("model_config"),
        "val_bleu": ckpt.get("val_bleu"),
    }
    torch.save(out, dst)
    print(f"Wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="checkpoint.pt", help="Full checkpoint from save_checkpoint")
    parser.add_argument("--dst", default="model_weights.pt", help="Output weights-only file")
    parser.add_argument("--fp16", action="store_true", help="Also save fp16 copy (smaller)")
    args = parser.parse_args()
    export_weights(args.src, args.dst)
    if args.fp16:
        base, ext = os.path.splitext(args.dst)
        export_weights(args.src, f"{base}_fp16{ext}", fp16=True)
