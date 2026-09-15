"""CLI to resolve Stack-Edu content from the Software Heritage S3 mirror
and cache it locally as an Arrow dataset (infra spec section 6.1).

Run this ONCE on the training cluster -- it needs AWS credentials
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) and a fast network link, and
performs millions of individual S3 fetches. Do not run this on a local
dev machine, and never invoke it as a side effect of the training loop.

Usage:
    python scripts/prepare_stack_edu.py --output-dir /data/stack_edu_prepared
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgealign.data import stack_edu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--min-int-score", type=int, default=3)
    args = parser.parse_args()

    stack_edu.prepare(args.output_dir, languages=args.languages, min_int_score=args.min_int_score)


if __name__ == "__main__":
    main()
