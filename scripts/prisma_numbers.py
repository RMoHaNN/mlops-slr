"""Print PRISMA flow numbers from the archived screening files.

Run after the pipeline (or on the archived CSVs) to reproduce the
numbers reported in the manuscript:

    python scripts/prisma_numbers.py
"""

import csv
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).parent.parent
SCREENING = PROJECT / "screening"

abs_rows = list(csv.DictReader(open(
    SCREENING / "abstract_screening.csv", encoding="utf-8")))
ft_rows = list(csv.DictReader(open(
    SCREENING / "fulltext_screening.csv", encoding="utf-8")))

print("=== IDENTIFICATION ===")
print(f"Unique records after deduplication: {len(abs_rows)}")
print(dict(Counter(r["stream"] for r in abs_rows)))

print("\n=== ABSTRACT SCREENING ===")
print(f"Total screened: {len(abs_rows)}")
print(dict(Counter(r["decision"] for r in abs_rows)))
print("Exclusion codes:",
      dict(Counter(r["exclusion_code"] for r in abs_rows
                   if r["decision"] == "exclude")))

passed = [r for r in abs_rows if r["decision"] in ("include", "borderline")]
print(f"Passed to full text: {len(passed)}",
      dict(Counter(r["stream"] for r in passed)))

print("\n=== FULL-TEXT SCREENING ===")
print(f"Total records: {len(ft_rows)}")
print(dict(Counter(r["decision"] for r in ft_rows)))
print("Exclusion codes:",
      dict(Counter(r["exclusion_code"] for r in ft_rows
                   if r["decision"] in ("exclude", "excluded"))))

included = [r for r in ft_rows if r["decision"] == "include"]
print(f"\nIncluded in synthesis: {len(included)}")
print(dict(Counter(r["stream"] for r in included)))

nopdf = sum(1 for r in ft_rows if r["exclusion_code"] == "FE_NOPDF")
content = sum(1 for r in ft_rows
              if r["decision"] in ("exclude", "excluded")
              and r["exclusion_code"] != "FE_NOPDF")
balance = len(passed) - nopdf - content
print(f"\nPRISMA balance: {len(passed)} passed - {nopdf} no full text "
      f"- {content} content-excluded = {balance} "
      f"({'BALANCED' if balance == len(included) else 'MISMATCH'})")
