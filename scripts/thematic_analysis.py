"""
Thematic analysis and cross-stream gap identification for the MLOps SLR.

Reads coded papers from fulltext_screening.csv, joins with stream info from
collection_mapping.csv, then runs three LLM passes:

  Pass 1 — Within-stream theme identification (separately for Generic and Industrial)
  Pass 2 — Cross-stream comparison: which industrial themes are absent/weak in generic?
  Pass 3 — Design principles: one principle per high/medium gap (answers RQ2)

Outputs:
  analysis/results/themes_generic.json
  analysis/results/themes_industrial.json
  analysis/results/cross_stream_gaps.json
  analysis/results/design_principles.json
  analysis/results/thematic_report.md   ← human-readable summary
"""

from __future__ import annotations
import csv, json, os, re, sys, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RESULTS = PROJECT / "analysis" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CONFIG  = Path.home() / ".config" / "academic-research" / "config.toml"

# ── Model selection ────────────────────────────────────────────────────────────
# Change to "claude-haiku-4-5-20251001" if using Anthropic
MODEL = "gemini-2.5-flash"


# ── API helpers ────────────────────────────────────────────────────────────────

def read_toml_value(text: str, section: str, key: str) -> str:
    m = re.search(rf'\[{section}\].*?{key}\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL)
    return m.group(1) if m else ""


def get_api_key(provider: str) -> str:
    env_var = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    if CONFIG.exists():
        return read_toml_value(CONFIG.read_text(encoding="utf-8"), provider, "api_key")
    return ""


def call_llm(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    """Call either Gemini or Claude depending on MODEL setting."""
    if MODEL.startswith("gemini"):
        return _call_gemini(prompt, system, max_tokens)
    elif MODEL.startswith("claude"):
        return _call_claude(prompt, system, max_tokens)
    raise ValueError(f"Unknown model prefix: {MODEL}")


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types
    api_key = get_api_key("gemini")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found")
    client = genai.Client(api_key=api_key)
    wait = 20
    for attempt in range(8):
        try:
            cfg = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            if system:
                cfg.system_instruction = system
            resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
            text = resp.text
            if text is None:
                for part in resp.candidates[0].content.parts:
                    if getattr(part, 'text', None) and not getattr(part, 'thought', False):
                        text = part.text
                        break
            return (text or "").strip()
        except Exception as e:
            if any(x in str(e).lower() for x in ("429", "quota", "rate", "resource exhausted",
                                                 "503", "unavailable", "overloaded", "500",
                                                 "internal", "timeout", "deadline")):
                print(f"  [rate limit] sleeping {wait}s …", flush=True)
                time.sleep(wait); wait = min(wait * 2, 120)
            else:
                raise
    raise RuntimeError("Rate limit persisted")


def _call_claude(prompt: str, system: str, max_tokens: int) -> str:
    import anthropic
    api_key = get_api_key("anthropic")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")
    client = anthropic.Anthropic(api_key=api_key)
    wait = 15
    for attempt in range(6):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=max_tokens, temperature=0.0,
                system=system or "You are a systematic review analyst.",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if "rate" in str(e).lower() or "529" in str(e):
                print(f"  [rate limit] sleeping {wait}s …", flush=True)
                time.sleep(wait); wait = min(wait * 2, 120)
            else:
                raise
    raise RuntimeError("Rate limit persisted")


def extract_json(text: str) -> dict | list:
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?', '', text)
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if not m:
        raise ValueError(f"No JSON found in response:\n{text[:300]}")
    return json.loads(m.group(1))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_coded_papers() -> list[dict]:
    seen: dict[str, dict] = {}
    with open(PROJECT / "screening" / "fulltext_screening.csv",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # last row per paper wins; papers without a DOI key on title
            key = r["doi"].strip().lower() or "title:" + r["title"].strip().lower()
            seen[key] = r
    papers = []
    for r in seen.values():
        if r.get("decision") != "include":
            continue
        if not r.get("stream"):
            r["stream"] = "unknown"
        papers.append(r)
    return papers


# ── Pass 1: Within-stream theme identification ─────────────────────────────────

THEME_SYSTEM = (
    "You are a qualitative researcher conducting thematic analysis of a systematic "
    "literature review on MLOps / AI implementation in production environments. "
    "You identify recurring themes from extracted paper codings. "
    "Return ONLY valid JSON — no prose."
)


def build_stream_text(papers: list[dict], fields: list[str]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        title = p.get("title", "")[:80]
        content = " | ".join(
            f"{f}: {p.get(f, '').strip()[:300]}"
            for f in fields if p.get(f, "").strip()
        )
        if content:
            lines.append(f"[{i}] {title}\n    {content}")
    return "\n\n".join(lines)


def identify_themes(papers: list[dict], stream_label: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Pass 1: Identifying themes — {stream_label} stream ({len(papers)} papers)")

    text = build_stream_text(papers, [
        "operational_constraints", "mlops_practices",
        "gaps_identified", "deployment_context"
    ])

    prompt = f"""You are analysing {len(papers)} papers from the {stream_label} AI implementation literature.

Below are the operational constraints, MLOps practices, gaps, and deployment contexts extracted from each paper.

{text}

TASK: Identify 6-10 major themes that recur across these papers. Themes should capture the STRUCTURAL characteristics of {stream_label} AI deployment, not just list topics.

For each theme return:
- id: "T{stream_label[:1]}1", "T{stream_label[:1]}2" etc.
- name: concise theme name (4-6 words)
- description: 2-3 sentences explaining the theme and why it matters
- sub_themes: list of 2-4 specific sub-aspects
- paper_count: approximate number of papers touching this theme
- representative_quotes: list of 2-3 direct quotes or close paraphrases from the constraints/gaps above
- constraint_type: "technical" | "organisational" | "regulatory" | "environmental" | "mixed"

Return JSON: {{"stream": "{stream_label}", "themes": [...]}}"""

    raw = call_llm(prompt, system=THEME_SYSTEM, max_tokens=8000)
    result = extract_json(raw)
    print(f"  ->{len(result.get('themes', []))} themes identified")
    return result


# ── Pass 2: Cross-stream comparison ───────────────────────────────────────────

GAP_SYSTEM = (
    "You are a comparative researcher identifying structural gaps between two "
    "bodies of literature on AI/ML deployment. Your goal is to find constraints "
    "that are UNIQUE TO or SIGNIFICANTLY STRONGER IN the Industrial stream — "
    "these gaps motivate design principles for Industrial AI. "
    "Return ONLY valid JSON."
)


def cross_stream_comparison(generic_themes: dict, industrial_themes: dict) -> dict:
    print(f"\n{'='*60}")
    print("Pass 2: Cross-stream gap analysis")

    g_themes = json.dumps(generic_themes.get("themes", []), indent=2)
    i_themes = json.dumps(industrial_themes.get("themes", []), indent=2)

    prompt = f"""You are comparing operational constraint themes across two MLOps literature streams.

STREAM A — Generic AI Implementation (cloud, SaaS, enterprise software):
{g_themes}

STREAM B — Industrial AI Implementation (manufacturing, Industry 4.0, predictive maintenance):
{i_themes}

TASK: For each INDUSTRIAL theme, assess whether it appears in the Generic literature.

Gap levels:
- HIGH: Theme is essentially absent from generic literature — industrial environments face this alone
- MEDIUM: Theme exists in generic literature but industrial context makes it fundamentally harder
- LOW: Theme is similar across both streams — generic practices largely transfer

For each industrial theme return:
- industrial_theme_id: the theme ID from stream B
- industrial_theme_name: the theme name
- present_in_generic: true | false | "partially"
- gap_level: "HIGH" | "MEDIUM" | "LOW"
- gap_explanation: 2-3 sentences explaining WHY this is a gap
- why_generic_falls_short: specific reason generic MLOps practices cannot address this
- closest_generic_theme: name of the closest generic theme, or null

Return JSON: {{"gaps": [...]}}"""

    raw = call_llm(prompt, system=GAP_SYSTEM, max_tokens=6000)
    result = extract_json(raw)
    gaps = result.get("gaps", [])
    high = sum(1 for g in gaps if g.get("gap_level") == "HIGH")
    medium = sum(1 for g in gaps if g.get("gap_level") == "MEDIUM")
    print(f"  ->{len(gaps)} comparisons: {high} HIGH gaps, {medium} MEDIUM gaps")
    return result


# ── Pass 3: Design principles ─────────────────────────────────────────────────

DP_SYSTEM = (
    "You are a research synthesiser deriving design principles from a systematic "
    "literature review. Design principles should be actionable, grounded in the "
    "evidence, and address a specific gap that generic MLOps cannot fill. "
    "Return ONLY valid JSON."
)


def generate_design_principles(gaps: dict, industrial_papers: list[dict]) -> dict:
    print(f"\n{'='*60}")
    print("Pass 3: Generating design principles (RQ2)")

    high_medium_gaps = [
        g for g in gaps.get("gaps", [])
        if g.get("gap_level") in ("HIGH", "MEDIUM")
    ]

    # Build evidence context from papers
    evidence = build_stream_text(industrial_papers[:30], ["gaps_identified", "key_findings"])

    prompt = f"""You are formulating design principles for Industrial AI Implementation based on a systematic literature review.

The following gaps were identified between Generic AI and Industrial AI MLOps practices (HIGH and MEDIUM gaps only):

{json.dumps(high_medium_gaps, indent=2)}

Supporting evidence from industrial papers:
{evidence[:3000]}

TASK: For each HIGH/MEDIUM gap, formulate ONE design principle.

A good design principle:
- Addresses a specific constraint that generic MLOps cannot solve
- Is actionable (tells practitioners what to do, not just what the problem is)
- Is grounded in the evidence from the papers
- Is distinct from the other principles (no overlap)

For each principle return:
- id: "DP-1", "DP-2" etc. (ordered by importance — HIGH gaps first)
- name: concise principle name (e.g. "OT-Aware Deployment Scheduling")
- statement: the principle in one sentence ("Industrial AI systems should...")
- rationale: 2-3 sentences explaining what gap this addresses and why
- implementation_guidance: 3-5 concrete implementation actions as a list
- supporting_gap: the industrial theme ID this addresses
- evidence_strength: "strong" | "moderate" | "emerging"

Return JSON: {{"research_question": "RQ2", "design_principles": [...]}}"""

    raw = call_llm(prompt, system=DP_SYSTEM, max_tokens=6000)
    result = extract_json(raw)
    dps = result.get("design_principles", [])
    print(f"  ->{len(dps)} design principles generated")
    return result


# ── Report generation ──────────────────────────────────────────────────────────

def write_report(
    generic_themes: dict,
    industrial_themes: dict,
    gaps: dict,
    design_principles: dict,
    papers: list[dict],
) -> None:
    generic_papers  = [p for p in papers if p["stream"] == "generic"]
    industrial_papers = [p for p in papers if p["stream"] == "industrial"]

    lines = [
        "# Thematic Analysis Report — MLOps / Industrial AI SLR",
        "",
        f"**Included papers:** {len(papers)} ({len(generic_papers)} generic, {len(industrial_papers)} industrial)",
        "",
        "---",
        "",
        "## RQ1: Operational Constraints",
        "",
        "### Generic AI Implementation — Themes",
        "",
    ]

    for t in generic_themes.get("themes", []):
        lines += [
            f"#### {t['id']}: {t['name']}",
            f"*{t.get('constraint_type','').capitalize()} constraint — {t.get('paper_count','')} papers*",
            "",
            t.get("description", ""),
            "",
            "**Sub-themes:** " + ", ".join(t.get("sub_themes", [])),
            "",
        ]

    lines += ["", "### Industrial AI Implementation — Themes", ""]
    for t in industrial_themes.get("themes", []):
        lines += [
            f"#### {t['id']}: {t['name']}",
            f"*{t.get('constraint_type','').capitalize()} constraint — {t.get('paper_count','')} papers*",
            "",
            t.get("description", ""),
            "",
            "**Sub-themes:** " + ", ".join(t.get("sub_themes", [])),
            "",
        ]

    lines += [
        "---",
        "",
        "## Cross-Stream Gap Analysis",
        "",
        "| Industrial Theme | Present in Generic? | Gap Level | Why Generic Falls Short |",
        "|-----------------|--------------------|-----------|-----------------------|",
    ]
    for g in gaps.get("gaps", []):
        present = str(g.get("present_in_generic", ""))
        lines.append(
            f"| {g.get('industrial_theme_name','')} "
            f"| {present} "
            f"| **{g.get('gap_level','')}** "
            f"| {g.get('why_generic_falls_short','')[:120]} |"
        )

    lines += [
        "",
        "---",
        "",
        "## RQ2: Design Principles for Industrial AI Implementation",
        "",
    ]
    for dp in design_principles.get("design_principles", []):
        lines += [
            f"### {dp['id']}: {dp['name']}",
            f"*Evidence strength: {dp.get('evidence_strength','')}*",
            "",
            f"**Principle:** {dp.get('statement','')}",
            "",
            f"**Rationale:** {dp.get('rationale','')}",
            "",
            "**Implementation guidance:**",
        ]
        for action in dp.get("implementation_guidance", []):
            lines.append(f"- {action}")
        lines.append("")

    report_path = RESULTS / "thematic_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading coded papers …")
    papers = load_coded_papers()
    generic_papers    = [p for p in papers if p["stream"] == "generic"]
    industrial_papers = [p for p in papers if p["stream"] == "industrial"]

    print(f"  Generic: {len(generic_papers)}  |  Industrial: {len(industrial_papers)}")

    if not papers:
        print("ERROR: No included papers found in fulltext_screening.csv")
        sys.exit(1)

    # Pass 1
    generic_themes    = identify_themes(generic_papers, "Generic")
    industrial_themes = identify_themes(industrial_papers, "Industrial")

    (RESULTS / "themes_generic.json").write_text(
        json.dumps(generic_themes, indent=2), encoding="utf-8")
    (RESULTS / "themes_industrial.json").write_text(
        json.dumps(industrial_themes, indent=2), encoding="utf-8")

    # Pass 2
    gaps = cross_stream_comparison(generic_themes, industrial_themes)
    (RESULTS / "cross_stream_gaps.json").write_text(
        json.dumps(gaps, indent=2), encoding="utf-8")

    # Pass 3
    design_principles = generate_design_principles(gaps, industrial_papers)
    (RESULTS / "design_principles.json").write_text(
        json.dumps(design_principles, indent=2), encoding="utf-8")

    # Report
    write_report(generic_themes, industrial_themes, gaps, design_principles, papers)

    print("\n" + "="*60)
    print("Analysis complete. Outputs in analysis/results/:")
    print("  themes_generic.json")
    print("  themes_industrial.json")
    print("  cross_stream_gaps.json")
    print("  design_principles.json")
    print("  thematic_report.md")


if __name__ == "__main__":
    main()
