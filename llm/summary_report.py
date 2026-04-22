"""
Generate evaluation metric report over all experiments in a given base directory.

Inputs: `out_dir` - experiment base directory.

Outputs:
- `.{out_dir}/summary_report.txt`
- `.{out_dir}/summary_report.csv`

"""

import argparse
import logging
from pathlib import Path

import polars as pl

from llm.evaluators import HumanVsLlmSummaryReport

# ---------- Logging ----------


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------- Entry Point ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for all experiments and produce a report in .csv and .txt format."
    )

    parser.add_argument(
        "--out_dir",
        "-od",
        default="llm/.scratch",
        help="Experiment output root folder.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("Generating summary report...")
    r = HumanVsLlmSummaryReport()
    report = r.generate(args.out_dir)
    csv_path = Path(args.out_dir) / "summary_report.csv"
    txt_path = Path(args.out_dir) / "summary_report.txt"
    with (
        txt_path.open("w", encoding="utf-8") as f,
        pl.Config(tbl_rows=-1, tbl_width_chars=10**3, set_fmt_str_lengths=50),
    ):
        f.write(report.__repr__())
    report.write_csv(csv_path, separator=",")
