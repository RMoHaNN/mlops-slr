# Scopus Search Strings

**Database:** Scopus  
**Date executed:** 25 May 2026  
**Time window:** PUBYEAR > 2017 (i.e. 2018–2026)

---

## Stream A — Generic MLOps (268 records)

```
TITLE-ABS-KEY ( ( "MLOps" OR "machine learning operations" ) AND ( "case study" OR "real-world" OR "in practice" OR "industrial application" ) ) AND PUBYEAR > 2017
```

Export file: `scopus_export_May 25-2026_f2f33b4c-1651-4a3d-8536-e4662d3266f5.csv`

---

## Stream B — Industrial AI (56 records)

```
TITLE-ABS-KEY ( ( "MLOps" OR "machine learning operations" ) AND ( "case study" OR "real-world" OR "in practice" OR "industrial application" ) AND ( "manufacturing" OR "predictive maintenance" OR "condition monitoring" OR "industrial" OR "Industry 4.0" ) ) AND PUBYEAR > 2017
```

Export file: `scopus_export_May 25-2026_9234a3e9-cbf1-472f-8b80-8eb18be4e56f.csv`

---

## Notes

- Stream B is a constrained subset of Stream A — it adds manufacturing/industrial terms to the same base query.
- 56 records appeared in both exports (Stream B ⊂ Stream A). These duplicates were removed from Stream A, leaving 212 unique Generic papers.
- Total unique corpus: 268 papers (212 Generic + 56 Industrial).
- The search intentionally targets implementation-oriented papers ("case study", "real-world", "in practice") to ensure the corpus reflects operational deployment rather than model development.
