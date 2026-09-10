"""Sync the AGY profile body from the organizational PEER contract (never edit upstream)."""

import argparse
from pathlib import Path

PROFILE = Path(__file__).resolve().parents[1] / ".agents/agents/slp-peer.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="Path to Paseo docs/slp-r1/roles/PEER.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    contract = args.contract.read_text(encoding="utf-8").strip() + "\n"
    if not contract.startswith("# SLP role contract: PEER\n"):
        parser.error("Expected the organizational PEER contract")
    profile = PROFILE.read_text(encoding="utf-8")
    header, _ = profile.split("\n---\n", 1)
    expected = header + "\n---\n\n" + contract
    if args.check:
        if profile != expected:
            parser.exit(1, "PEER profile is out of sync\n")
        print("PEER contract matches")
    else:
        PROFILE.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
