"""Fetch a Qwen checkpoint to a local directory.

Downloads config, tokenizer and safetensors shards so the LLM experiments can
load from disk rather than the Hub. GGUF files are skipped.

The default repo is `Qwen/Qwen3-4B`, a dense checkpoint that loads with
AutoModelForSequenceClassification under transformers>=4.51. MoE and multimodal
Qwen variants are not supported by this pipeline.

Run:
    python download_qwen.py                              # -> models/qwen3_4b
    python download_qwen.py --repo-id Qwen/Qwen3-8B
    python download_qwen.py --dest /path/on/scratch      # set LLM_CLS_MODEL to match

For faster downloads, pip install "huggingface_hub[hf_transfer]" and export
HF_HUB_ENABLE_HF_TRANSFER=1. For a gated repo, pass --hf-token or set HF_TOKEN.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Destination matches where transfer_llm_cls looks when neither --model nor
# $LLM_CLS_MODEL is set.
DEFAULT_REPO_ID = "Qwen/Qwen3-4B"
DEFAULT_DEST = str(Path(__file__).resolve().parent / "models" / "qwen3_4b")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Qwen checkpoint to a local directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help=f"Hugging Face repo id (default: {DEFAULT_REPO_ID}).")
    parser.add_argument("--dest", type=Path, default=Path(DEFAULT_DEST),
                        help=f"Local directory to download into (default: {DEFAULT_DEST}).")
    parser.add_argument("--revision", default=None,
                        help="Optional git revision / tag / commit to pin.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HF token for gated repos (or set HF_TOKEN env var).")
    parser.add_argument("--ignore-patterns", nargs="+", default=["*.gguf"],
                        help="Globs to skip (default skips GGUF).")
    parser.add_argument("--allow-patterns", nargs="+", default=None,
                        help="If set, download ONLY files matching these globs.")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{args.repo_id}' -> {args.dest}")
    if args.revision:
        print(f"  revision: {args.revision}")
    print(f"  ignore: {args.ignore_patterns}  allow: {args.allow_patterns}")

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.dest),
        revision=args.revision,
        token=args.hf_token,
        ignore_patterns=args.ignore_patterns,
        allow_patterns=args.allow_patterns,
    )

    print(f"\nDone. Model files at: {path}")
    # Check the files the loader needs are present.
    needed = ["config.json", "tokenizer_config.json"]
    safetensors = list(args.dest.glob("*.safetensors"))
    for f in needed:
        print(f"  {f}: {'found' if (args.dest / f).exists() else 'MISSING'}")
    print(f"  *.safetensors shards: {len(safetensors)} "
          f"{'found' if safetensors else 'MISSING — check the repo / patterns'}")
    print(f"\nPoint the experiments at it:")
    print(f"  export LLM_CLS_MODEL={args.dest}")


if __name__ == "__main__":
    main()