"""Convert databricks/databricks-dolly-15k to RWKV-7 chat JSONL for FlowTrain SFT.

Output records use the ``{"prompt", "completion"}`` shape that
``flowtrain.sft_data.SFTJsonlDataset`` masks the prompt side of (loss on the
assistant completion only). The RWKV-7 chat template is applied here, not the
Alapaca one, following the official RWKV prompt guide:

- Chat-round separator is ``"\\n\\n"``. Any ``"\\n\\n"`` *inside* user content
  (instruction / context / response) is collapsed to ``"\\n"`` so it is not
  mistaken for a round boundary.
- The prompt ends at ``Assistant:`` with NO trailing whitespace (a trailing
  space upsets the RWKV tokenizer and degrades output), so the prompt is
  ``.rstrip()``-ed.
- The completion begins with a single space (``" Assistant: <text>"``) and ends
  with ``"\\n\\n"`` so the model learns to stop after its turn. The tokenizer
  adapter has ``eos_token_id=None``, so SFTJsonlDataset's ``add_eos`` is a
  no-op and termination rides on this ``"\\n\\n"``.

dolly has no system field, so no ``System:`` line is emitted.

Usage::

    python scripts/convert_dolly_to_rwkv_jsonl.py --out data/dolly-rwkv.jsonl --limit 800
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collapse_round_sep(text: str) -> str:
    """``"\\n\\n"`` is the RWKV chat-round separator; collapse it inside content."""
    return text.replace("\r\n", "\n").replace("\n\n", "\n")


def build_prompt(instruction: str, context: str) -> str:
    instruction = collapse_round_sep(instruction).strip()
    context = collapse_round_sep(context).strip()
    if context:
        prompt = f"User: {instruction}\n{context}\n\nAssistant:"
    else:
        prompt = f"User: {instruction}\n\nAssistant:"
    return prompt.rstrip()


def build_completion(response: str) -> str:
    response = collapse_round_sep(response).strip()
    return f" {response}\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert databricks-dolly-15k to RWKV-7 chat JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k", help="HF dataset id")
    parser.add_argument("--split", default="train", help="Dataset split to convert")
    parser.add_argument("--limit", type=int, default=None, help="Only emit the first N records (smoke runs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("datasets is required: pip install datasets") from exc

    ds = load_dataset(args.dataset, split=args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            context = str(row.get("context", "") or "").strip()
            if not instruction or not response:
                skipped += 1
                continue
            record = {
                "prompt": build_prompt(instruction, context),
                "completion": build_completion(response),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.limit is not None and written >= args.limit:
                break

    print(f"wrote {written} records to {out_path} (skipped {skipped} empty)", file=sys.stderr)


if __name__ == "__main__":
    main()
