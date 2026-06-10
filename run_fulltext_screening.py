"""Stage 2 — full-text screening and coding for the MLOps / Industrial AI SLR.

For every paper that passed abstract screening (include or borderline),
looks up its PDF in output/pdf_cache/, extracts text, and asks Gemini to
confirm inclusion and extract all 10 coding fields.

Output: screening/fulltext_screening.csv  (resumable, one row per record)
"""

from __future__ import annotations

import csv
import json
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
    FULLTEXT_CODING_FIELDS,
    FULLTEXT_CODING_MODEL,
    FULLTEXT_CODING_PROMPT_VERSION,
    FULLTEXT_CODING_SYSTEM_PROMPT,
)

PROJECT = Path(__file__).parent
PDF_CACHE = PROJECT / "output" / "pdf_cache"
ABS_CSV = PROJECT / "screening" / "abstract_screening.csv"
OUT_CSV = PROJECT / "screening" / "fulltext_screening.csv"

MAX_CHARS = 700_000

CODING_FIELD_NAMES = [f["name"] for f in FULLTEXT_CODING_FIELDS]

COLUMNS = [
    "timestamp", "doi", "title", "year", "journal", "stream",
    "pdf_available", "fulltext_chars", "truncated",
    "decision", "exclusion_code", "reason",
    *CODING_FIELD_NAMES,
    "model", "prompt_version",
]


def get_gemini_key() -> str:
    text = (Path.home() / ".config" / "academic-research" / "config.toml"
            ).read_text(encoding="utf-8")
    m = re.search(r'\[gemini\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
    if not m:
        raise RuntimeError("Gemini api_key not found in config.toml")
    return m.group(1)


def record_key(doi: str, title: str) -> str:
    doi = doi.strip().lower()
    return doi if doi else "title:" + title.strip().lower()


def cache_filename(doi: str, title: str) -> str:
    """Cache filename: DOI-based, or a title slug for papers without a DOI
    (same convention as scripts/ingest_pdfs.py)."""
    doi = doi.strip()
    if doi:
        return doi.replace("/", "_") + ".pdf"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
    return "no-doi_" + slug + ".pdf"


def build_system_prompt() -> str:
    fields_placeholder = ", ".join(
        f'"{f["name"]}": "<extracted text>"' for f in FULLTEXT_CODING_FIELDS
    )
    return FULLTEXT_CODING_SYSTEM_PROMPT.replace(
        "{coding_fields_json_placeholder}", fields_placeholder
    )


def extract_pdf_text(path: Path) -> str | None:
    """Try pdfplumber first, then pypdf. Returns None if both fail."""
    try:
        import pdfplumber
        chunks = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
        text = "\n".join(chunks).strip()
        if text:
            return text
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text
    except Exception:
        pass
    return None


def call_gemini(client: genai.Client, system_prompt: str, user_msg: str) -> str:
    wait = 30
    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=FULLTEXT_CODING_MODEL,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    system_instruction=system_prompt,
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


def parse_json_response(raw: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", raw)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    with open(ABS_CSV, encoding="utf-8", newline="") as f:
        abs_rows = list(csv.DictReader(f))
    todo = [r for r in abs_rows if r["decision"] in ("include", "borderline")]
    n_inc = sum(1 for r in todo if r["decision"] == "include")
    n_bor = sum(1 for r in todo if r["decision"] == "borderline")
    print(f"Abstract passed: {n_inc} include + {n_bor} borderline = {len(todo)} to process")

    done: set[str] = set()
    file_exists = OUT_CSV.exists()
    if file_exists:
        with open(OUT_CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                # FE_NOPDF rows are reprocessed if the PDF has since been
                # added to the cache (a fresh row is appended; downstream
                # analysis keeps the last row per DOI).
                if r.get("exclusion_code") == "FE_NOPDF":
                    if (PDF_CACHE / cache_filename(r["doi"], r["title"])).exists():
                        continue
                if r.get("decision") == "error":
                    continue  # retry errors (downstream keeps last row per DOI)
                done.add(record_key(r["doi"], r["title"]))
    remaining = sum(1 for r in todo if record_key(r["doi"], r["title"]) not in done)
    print(f"Resuming: {len(done)} done, {remaining} remaining")

    client = genai.Client(api_key=get_gemini_key())
    system_prompt = build_system_prompt()

    out = open(OUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out, fieldnames=COLUMNS)
    if not file_exists:
        writer.writeheader()
        out.flush()

    total = len(todo)
    for i, r in enumerate(todo, 1):
        doi = r["doi"].strip()
        title = r["title"].strip()
        if record_key(doi, title) in done:
            continue
        print(f"Processing {i}/{total}: {title[:70]}...")

        row = {
            "timestamp": "",
            "doi": doi,
            "title": title,
            "year": r.get("year", ""),
            "journal": r.get("journal", ""),
            "stream": r.get("stream", ""),
            "pdf_available": "no",
            "fulltext_chars": "",
            "truncated": "",
            "decision": "",
            "exclusion_code": "",
            "reason": "",
            **{name: "" for name in CODING_FIELD_NAMES},
            "model": FULLTEXT_CODING_MODEL,
            "prompt_version": FULLTEXT_CODING_PROMPT_VERSION,
        }

        pdf_path = PDF_CACHE / cache_filename(doi, title)
        if not pdf_path.exists():
            print("  PDF: not found -> FE_NOPDF")
            row.update(
                decision="excluded",
                exclusion_code="FE_NOPDF",
                reason="Full text not available: PDF could not be obtained "
                       "(institutional access required, or paper is a book "
                       "chapter / edited volume without a standalone PDF)",
            )
        else:
            row["pdf_available"] = "yes"
            text = extract_pdf_text(pdf_path)
            if text is None:
                print("  PDF: text extraction failed")
                row.update(decision="error", reason="PDF text extraction failed")
            else:
                if len(text) > MAX_CHARS:
                    text = text[:MAX_CHARS]
                    row["truncated"] = "yes"
                else:
                    row["truncated"] = "no"
                row["fulltext_chars"] = str(len(text))
                print(f"  PDF: found ({len(text)} chars)")

                user_msg = f"Title: {title}\n\nFull text:\n{text}"
                raw = call_gemini(client, system_prompt, user_msg)
                parsed = parse_json_response(raw)
                if parsed is None:
                    row.update(decision="error",
                               reason="JSON parse failed: " + raw[:200])
                else:
                    row["decision"] = str(parsed.get("decision", "")).strip()
                    row["exclusion_code"] = str(parsed.get("exclusion_code", "")).strip()
                    row["reason"] = str(parsed.get("reason", "")).strip()
                    for name in CODING_FIELD_NAMES:
                        val = parsed.get(name, "")
                        if not isinstance(val, str):
                            val = json.dumps(val, ensure_ascii=False)
                        row[name] = val
                time.sleep(2)

        if row["decision"] == "include":
            print("  -> include")
        elif row["decision"] in ("exclude", "excluded"):
            print(f"  -> exclude ({row['exclusion_code']})")
        else:
            print(f"  -> {row['decision']}")

        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        writer.writerow(row)
        out.flush()
        done.add(record_key(doi, title))

    out.close()

    # Summary
    with open(OUT_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    included = sum(1 for r in rows if r["decision"] == "include")
    nopdf = sum(1 for r in rows if r["exclusion_code"] == "FE_NOPDF")
    content_excl = sum(
        1 for r in rows
        if r["decision"] in ("exclude", "excluded") and r["exclusion_code"] != "FE_NOPDF"
    )
    errors = sum(1 for r in rows if r["decision"] == "error")
    print("\n=== Full-text screening summary ===")
    print(f"included {included}, excluded {content_excl} (content), "
          f"excluded {nopdf} (FE_NOPDF), error {errors}")


if __name__ == "__main__":
    main()
