"""Download the Qwen3-4B checkpoint to models/qwen3_4b for the LLM classifiers.

    python download_qwen.py [--dest DIR]   # then export LLM_CLS_MODEL=DIR
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Same default location transfer_llm_cls loads from.
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
                        help="Git revision, tag or commit to pin.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HF token for gated repos (default: $HF_TOKEN).")
    parser.add_argument("--ignore-patterns", nargs="+", default=["*.gguf"],
                        help="Globs to skip.")
    parser.add_argument("--allow-patterns", nargs="+", default=None,
                        help="If set, download only files matching these globs.")
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