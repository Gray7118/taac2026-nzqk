"""Patch schema.json with correct ts_fid values to activate time features.

Reads a schema.json (from the platform-provided data directory), auto-detects
the timestamp column in each sequence domain by vocabulary size, sets the
``ts_fid`` field, and writes the patched schema to a separate output file.

Detection strategy: in each domain's feature list, the timestamp column has a
vocabulary size of ~1.77 billion — far larger than any genuine categorical
feature (the next-largest is ~278 million).  All features with vocab greater
than ``TIMESTAMP_VOCAB_THRESHOLD`` (1 billion) are identified as timestamp
columns.

This approach is domain-name-agnostic and does not depend on any hardcoded fid
mapping, making it robust across different platform deployments.

The original schema.json is never modified — this script always writes to a
new path, making it safe for read-only platform environments.

Usage:
    python3 fix_schema.py <input_schema.json> <output_schema.json>

On the competition platform this is called from run.sh before training.
"""

import json
import os
import sys

# Features with vocab > 1B are timestamp columns (Unix seconds).
# The largest genuine categorical feature across all domains is ~278M
# (domain_c seq_47), giving a clear separation margin.
TIMESTAMP_VOCAB_THRESHOLD = 1_000_000_000


def _find_ts_candidate(features: list, current_ts_fid):
    """Return ``(fid, vocab_size)`` for a timestamp candidate, or ``None``.

    If ``current_ts_fid`` is already set and points to a threshold-crossing
    feature, returns ``(current_ts_fid, vocab_size)`` (signalling idempotent).

    Otherwise scans ``features`` for the entry with the highest
    threshold-crossing vocab size.
    """
    ts_entries = [(fid, vs) for fid, vs in features if vs > TIMESTAMP_VOCAB_THRESHOLD]

    if not ts_entries:
        return None

    # If the currently-set ts_fid is one of the candidates, it's already correct.
    if current_ts_fid is not None:
        for fid, vs in ts_entries:
            if fid == current_ts_fid:
                return (fid, vs)

    # Pick the one with the highest vocab (least ambiguous timestamp).
    ts_entries.sort(key=lambda x: x[1], reverse=True)
    return ts_entries[0]


def detect_and_set_ts_fid(schema: dict) -> int:
    """Auto-detect timestamp features in every sequence domain by vocab size.

    For each domain under ``schema["seq"]``, scans the features list and
    sets ``ts_fid`` to the fid of the feature whose vocab_size exceeds
    ``TIMESTAMP_VOCAB_THRESHOLD``.

    If a domain already has a correctly-set ``ts_fid``, it is left unchanged
    (idempotent).

    Args:
        schema: The loaded schema dict (mutated in-place).

    Returns:
        Number of domains patched.
    """
    if "seq" not in schema:
        print("[fix_schema] WARNING: no 'seq' key in schema, skipping")
        return 0

    patched = 0

    for domain_name, cfg in schema["seq"].items():
        features = cfg.get("features", [])
        old_ts_fid = cfg.get("ts_fid")

        candidate = _find_ts_candidate(features, old_ts_fid)

        if candidate is None:
            print(
                f"[fix_schema] {domain_name!r}: "
                f"no timestamp feature found (all vocab < {TIMESTAMP_VOCAB_THRESHOLD}), "
                f"keeping ts_fid={old_ts_fid}"
            )
            continue

        new_fid, vs = candidate

        if new_fid == old_ts_fid:
            print(
                f"[fix_schema] {domain_name!r}: "
                f"ts_fid already = {old_ts_fid} (vocab={vs}), no change"
            )
            continue

        if old_ts_fid is not None:
            print(
                f"[fix_schema] {domain_name!r}: "
                f"ts_fid was {old_ts_fid} but vocab looks non-timestamp, "
                f"overriding → {new_fid} (vocab={vs})"
            )
        else:
            print(
                f"[fix_schema] {domain_name!r}: "
                f"ts_fid {old_ts_fid} → {new_fid} (vocab={vs})"
            )

        cfg["ts_fid"] = new_fid
        patched += 1

    return patched


def patch_schema(input_path: str, output_path: str) -> None:
    """Patch the schema and write it to output_path.

    Args:
        input_path: Path to the original schema.json (read-only).
        output_path: Path where the patched schema will be written.

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    patched = detect_and_set_ts_fid(schema)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print(f"[fix_schema] Patched {patched} domains, wrote to {output_path}")


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <input_schema.json> <output_schema.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"[fix_schema] ERROR: {input_path} not found")
        sys.exit(1)

    patch_schema(input_path, output_path)


if __name__ == "__main__":
    main()