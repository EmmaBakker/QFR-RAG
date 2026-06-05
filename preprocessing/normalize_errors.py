#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple


class TextNormalizer:
    def __init__(self, replacements: Dict[str, str]):
        self.replacements = replacements

    @classmethod
    def from_json(cls, path: str | Path) -> "TextNormalizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        reps = data.get("replacements", {})
        if not isinstance(reps, dict) or not reps:
            raise ValueError("JSON must contain a non-empty 'replacements' object.")
        reps = {str(k): str(v) for k, v in reps.items()}
        return cls(replacements=reps)

    def normalize(self, text: str) -> Tuple[str, Dict[str, int]]:
        out = text or ""
        stats: Dict[str, int] = {}
        # longest keys first to avoid partial overlaps
        for src in sorted(self.replacements.keys(), key=len, reverse=True):
            dst = self.replacements[src]
            if src and src in out:
                count = out.count(src)
                out = out.replace(src, dst)
                stats[src] = count
        return out, stats


# Config
BLOCKS_DIR = Path("data/processed")
MAPPING_PATH = Path("preprocessing/text_normalization.json")
INPLACE = True


def main() -> None:
    normalizer = TextNormalizer.from_json(MAPPING_PATH)

    jsonl_files = sorted(BLOCKS_DIR.rglob("*.blocks.jsonl"))

    if not jsonl_files:
        raise RuntimeError(f"No *.blocks.jsonl or blocks.json files found under {BLOCKS_DIR.resolve()}")

    if jsonl_files:
        print(f"Found {len(jsonl_files)} *.blocks.jsonl files in {BLOCKS_DIR}")
        for path in jsonl_files:
            total_changes = 0
            out_lines = []

            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                new_text, stats = normalizer.normalize(text)
                if stats:
                    obj["text"] = new_text
                    obj["char_len"] = len(new_text)
                    obj["text_hash"] = sha1_text(new_text)
                    total_changes += sum(stats.values())
                out_lines.append(json.dumps(obj, ensure_ascii=False))

            if total_changes == 0:
                continue

            out_path = path if INPLACE else path.with_suffix(".normalized.jsonl")
            out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            print(f"{path.name}: {total_changes} replacements")

    print("Done.")


def sha1_text(s: str) -> str:
    import hashlib
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
