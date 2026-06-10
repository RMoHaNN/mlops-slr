"""Stage 1 — abstract screening for the MLOps / Industrial AI SLR.

Reads both Scopus exports, deduplicates Stream B into Stream A (Stream B
is a topical subset), tags every record with its stream, pre-screens by
document type, then screens the rest with Gemini on title + abstract.

Output: screening/abstract_screening.csv  (resumable, one row per record)
"""

from __future__ import annotations

import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from google import genai
from google.genai import types

from screening_config import (
    ABSTRACT_SCREENING_MODEL,
    ABSTRACT_SCREENING_PROMPT_VERSION,
    ABSTRACT_SCREENING_SYSTEM_PROMPT,
)

PROJECT = Path(__file__).parent
RAW = PROJECT / "data" / "raw"
STREAM_A_CSV = RAW / "scopus_export_May 25-2026_f2f33b4c-1651-4a3d-8536-e4662d3266f5.csv"
STREAM_B_CSV = RAW / "scopus_export_May 25-2026_9234a3e9-cbf1-472f-8b80-8eb18be4e56f.csv"
OUT_CSV = PROJECT / "screening" / "abstract_screening.csv"

COLUMNS = [
    "timestamp", "doi", "title", "year", "journal", "stream", "document_type",
    "decision", "exclusion_code", "reason", "model", "prompt_version",
]

# Document types excluded before any LLM call
PRESCREEN_EXCLUDE_TYPES = {"Book", "Conference review"}


def get_gemini_key() -> str:
    text = (Path.home() / ".config" / "academic-research" / "config.toml"
            ).read_text(encoding="utf-8")
    m = re.search(r'\[gemini\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
    if not m:
        raise RuntimeError("Gemini api_key not found in config.toml")
    return m.group(1)


def record_key(doi: str, title: str) -> str:
    """Stable identity for resume: DOI when present, else normalised title."""
    doi = doi.strip().lower()
    return doi if doi else "title:" + title.strip().lower()


def load_records() -> list[dict]:
    """Load both exports, dedupe B into A, tag stream."""
    with open(STREAM_A_CSV, encoding="utf-8-sig") as f:
        stream_a = list(csv.DictReader(f))
    with open(STREAM_B_CSV, encoding="utf-8-sig") as f:
        stream_b = list(csv.DictReader(f))

    b_dois = {r["DOI"].strip().lower() for r in stream_b if r["DOI"].strip()}
    b_eids = {r.get("EID", "").strip() for r in stream_b if r.get("EID", "").strip()}

    records = []
    a_dois: set[str] = set()
    a_eids: set[str] = set()
    for r in stream_a:
        doi = r["DOI"].strip().lower()
        eid = r.get("EID", "").strip()
        if doi:
            a_dois.add(doi)
        if eid:
            a_eids.add(eid)
        industrial = (doi and doi in b_dois) or (eid and eid in b_eids)
        r["stream"] = "industrial" if industrial else "generic"
        records.append(r)

    # No-DOI edge case: append any Stream B record not present in Stream A
    appended = 0
    for r in stream_b:
        doi = r["DOI"].strip().lower()
        eid = r.get("EID", "").strip()
        if doi and doi in a_dois:
            continue
        if eid and eid in a_eids:
            continue
        r["stream"] = "industrial"
        records.append(r)
        appended += 1

    generic = sum(1 for r in records if r["stream"] == "generic")
    industrial = sum(1 for r in records if r["stream"] == "industrial")
    print(f"Generic: {generic}  Industrial: {industrial}  Total: {len(records)}")
    if appended:
        print(f"  (appended {appended} Stream B records not found in Stream A)")
    return records


def call_gemini(client: genai.Client, user_msg: str) -> str:
    wait = 30
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=ABSTRACT_SCREENING_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    system_instruction=ABSTRACT_SCREENING_SYSTEM_PROMPT,
                ),
            )
            text = resp.text
            if text is None:
                for part in resp.candidates[0].content.parts:
                    if getattr(part, "text", None) and not getattr(part, "thought", False):
                        text = part.text
                        break
            return (text or "").strip()
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ("429", "quota", "rate", "resource exhausted",
                                      "503", "unavailable", "overloaded", "500",
                                      "internal", "timeout", "deadline")):
                if attempt == 5:
                    raise
                print(f"  [rate limit] waiting {wait}s ...", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise
    raise RuntimeError("Rate limit persisted after retries")


def parse_response(text: str) -> tuple[str, str, str]:
    """Returns (decision, exclusion_code, reason)."""
    dm = re.search(r"DECISION:\s*(include|borderline|exclude)", text, re.IGNORECASE)
    rm = re.search(r"REASON:\s*(.+)", text)
    if not dm:
        return "error", "", text[:200]
    decision = dm.group(1).lower()
    reason = rm.group(1).strip() if rm else ""
    exclusion_code = ""
    if decision == "exclude":
        cm = re.search(r"\b(E[1-5])\b", reason)
        exclusion_code = cm.group(1) if cm else ""
    return decision, exclusion_code, reason


def main() -> None:
    records = load_records()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Counter, not set: the exports contain duplicate-titled no-DOI volume
    # records that must each get their own output row.
    from collections import Counter
    done: Counter = Counter()
    file_exists = OUT_CSV.exists()
    if file_exists:
        with open(OUT_CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                done[record_key(r["doi"], r["title"])] += 1
    n_done = sum(done.values())
    print(f"Resuming: {n_done} done, {len(records) - n_done} remaining")
    consumed: Counter = Counter()

    client = genai.Client(api_key=get_gemini_key())

    out = open(OUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out, fieldnames=COLUMNS)
    if not file_exists:
        writer.writeheader()
        out.flush()

    total = len(records)
    for i, r in enumerate(records, 1):
        title = r["Title"].strip()
        doi = r["DOI"].strip()
        k = record_key(doi, title)
        if consumed[k] < done[k]:
            consumed[k] += 1
            continue
        print(f"Processing {i}/{total}: {title[:70]}...")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "doi": doi,
            "title": title,
            "year": r.get("Year", "").strip(),
            "journal": r.get("Source title", "").strip(),
            "stream": r["stream"],
            "document_type": r.get("Document Type", "").strip(),
            "model": ABSTRACT_SCREENING_MODEL,
            "prompt_version": ABSTRACT_SCREENING_PROMPT_VERSION,
        }

        doc_type = row["document_type"]
        if doc_type in PRESCREEN_EXCLUDE_TYPES:
            row.update(
                decision="exclude",
                exclusion_code="E_DOC",
                reason="E_DOC: Not an individual research paper "
                       "(book or conference proceedings volume)",
            )
        else:
            abstract = r.get("Abstract", "").strip()
            if abstract and abstract.lower() != "[no abstract available]":
                user_msg = f"Title: {title}\nAbstract: {abstract}"
            else:
                user_msg = f"Title: {title}\n[No abstract available]"
            text = call_gemini(client, user_msg)
            decision, exclusion_code, reason = parse_response(text)
            row.update(decision=decision, exclusion_code=exclusion_code, reason=reason)
            time.sleep(1)

        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        writer.writerow(row)
        out.flush()
        consumed[k] += 1
        done[k] += 1

    out.close()

    # Summary
    with open(OUT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    counts = Counter(r["decision"] for r in rows)
    print("\n=== Abstract screening summary ===")
    print(f"Total rows: {len(rows)}")
    for decision, n in sorted(counts.items()):
        print(f"  {decision}: {n}")


if __name__ == "__main__":
    main()
