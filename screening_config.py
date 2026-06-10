"""Per-project screening configuration -- MLOps / Industrial AI SLR.

Compares Generic AI Implementation vs Industrial AI Implementation to
identify structural gaps that motivate design principles for deploying
and operating AI in industrial/manufacturing environments.

RQ1: What operational constraints distinguish Industrial AI implementation
     from Generic AI implementation?
RQ2: What design principles for Industrial AI can be identified from the
     gap between the two literatures?
"""

# =============================================================================
# Abstract screening (stage 1) -- Gemini 2.5 Flash on title + abstract
# =============================================================================

ABSTRACT_SCREENING_MODEL = "gemini-2.5-flash"
ABSTRACT_SCREENING_PROMPT_VERSION = "v1-2026-05-27"


ABSTRACT_SCREENING_SYSTEM_PROMPT = (
    "You are a systematic review screener for a literature review comparing "
    "Generic AI Implementation and Industrial AI Implementation.\n\n"
    "RESEARCH QUESTIONS:\n"
    "RQ1: What operational constraints distinguish Industrial AI implementation "
    "(manufacturing/maintenance environments) from Generic AI implementation "
    "(standard software-centric ML deployment contexts)?\n"
    "RQ2: What design principles for Industrial AI can be identified from the gap "
    "between the two literatures?\n\n"
    "A paper is relevant if it meets BOTH of the following criteria:\n\n"
    "1. DEPLOYMENT/OPERATIONS FOCUS: The paper addresses deploying, operating, "
    "monitoring, maintaining, or governing ML/AI models in production environments. "
    "Examples: MLOps pipelines, model monitoring, drift detection, retraining "
    "strategies, CI/CD for ML, model governance, production deployment architectures. "
    "NOT relevant: papers about ML model development, training, or architecture "
    "design only, with no discussion of how the model is operated after deployment. "
    "NOT relevant: papers using AI/ML as a black-box tool to solve a domain problem "
    "(e.g. applied a CNN to detect faults) without discussing operational aspects.\n\n"
    "2. DEPLOYMENT CONTEXT (either A or B):\n"
    "   A -- Industrial/Manufacturing AI: AI/ML deployment in manufacturing, "
    "Industry 4.0, predictive maintenance, industrial IoT, cyber-physical systems, "
    "process industries, energy/utilities, or safety-critical physical environments.\n"
    "   B -- Generic AI Implementation: AI/ML deployment in cloud-native, SaaS, "
    "enterprise software, data platforms, or general engineering contexts not "
    "tied to physical production processes.\n\n"
    "DECISION RULES:\n"
    "- INCLUDE: the abstract clearly addresses criterion 1 AND criterion 2.\n"
    "- EXCLUDE: the abstract clearly fails criterion 1 or criterion 2. Use these codes:\n"
    "  E1: No ML/AI component -- about automation, robotics, IoT, or IT systems "
    "without a machine learning or AI model\n"
    "  E2: ML/AI development/training only -- addresses model building, algorithms, "
    "or architecture but not deployment, monitoring, or production operations\n"
    "  E3: AI-as-tool without operational focus -- applies AI to solve a domain "
    "problem but does not discuss how the model is deployed or maintained\n"
    "  E4: Non-English language\n"
    "  E5: Inaccessible -- no abstract and title gives no basis for inclusion\n"
    "- BORDERLINE: abstract is missing or too vague to decide; title suggests "
    "relevance; paper sits at the boundary of development and deployment.\n\n"
    "BIAS: Be liberal. When uncertain between include and borderline, choose include. "
    "When uncertain between borderline and exclude, choose borderline. This review "
    "compares two literatures -- missing a paper from one stream weakens the comparison.\n\n"
    "Respond with EXACTLY two lines:\n"
    "DECISION: include|borderline|exclude\n"
    "REASON: <one sentence citing which criterion or exclusion code triggered the decision>"
)


# =============================================================================
# Full-text coding (stage 2) -- Gemini 2.5 Flash on full PDF text
# =============================================================================

FULLTEXT_CODING_MODEL = "gemini-2.5-flash"
FULLTEXT_CODING_PROMPT_VERSION = "v3-2026-06-03"

FULLTEXT_CODING_FIELDS = [
    {
        "name": "paper_type",
        "description": (
            "Classify the paper's primary contribution type. Choose one: "
            "architecture -- proposes or evaluates an MLOps reference architecture, "
            "platform design, or system blueprint; "
            "case_study -- reports a real-world deployment with empirical evidence "
            "from an actual organisation or system; "
            "empirical_study -- survey, interview, or experiment grounded in real "
            "practitioner data but not a single deployment story; "
            "conceptual -- proposes a framework, taxonomy, or design principles "
            "without empirical grounding in a real deployment; "
            "survey -- systematic or structured review of existing literature or tools."
        ),
    },
    {
        "name": "evidence_level",
        "description": (
            "How grounded is the paper's evidence? Choose one: "
            "real_deployment -- the paper reports results from a production system "
            "actually running in an organisation (not a lab or toy dataset); "
            "prototype -- implemented and evaluated on a realistic but non-production "
            "system, benchmark dataset, or controlled experiment; "
            "theoretical -- no implementation or only illustrative examples provided."
        ),
    },
    {
        "name": "deployment_context",
        "description": (
            "The specific deployment environment described in this paper. "
            "Choose the best-fit label and add a one-sentence clarification: "
            "cloud/SaaS | enterprise IT | industrial manufacturing | "
            "predictive maintenance | process industry | edge/IoT | "
            "safety-critical physical environment | multi-context | other."
        ),
    },
    {
        "name": "operational_constraints",
        "description": (
            "Constraints and challenges the paper identifies for deploying or "
            "operating AI in this specific context. Extract as a bulleted list "
            "of concrete constraints (e.g. OT/IT integration gaps, sensor drift, "
            "safety certification requirements, latency limits, scheduled "
            "downtime windows, data labelling scarcity, regulatory compliance, "
            "limited connectivity). Quote or closely paraphrase from the paper. "
            "This field directly answers RQ1."
        ),
    },
    {
        "name": "mlops_practices",
        "description": (
            "Specific MLOps or AI operations practices, tools, architectures, "
            "or processes described or proposed in this paper. Include: "
            "deployment strategies (canary, shadow, blue-green), monitoring "
            "approaches, drift detection methods, retraining triggers, "
            "pipeline automation, model registry use, versioning approaches, "
            "governance mechanisms, or any novel operational procedure. "
            "Be concrete -- name the practice and how it is applied."
        ),
    },
    {
        "name": "gaps_identified",
        "description": (
            "Gaps in current MLOps/AI operations practice that this paper "
            "explicitly acknowledges. These may appear as limitations, "
            "as problems the paper partially addresses, or as statements "
            "about what current tools/frameworks cannot handle. "
            "This field directly feeds into RQ2 -- gaps in the industrial "
            "stream that do not appear in the generic stream become "
            "motivation for design principles."
        ),
    },
    {
        "name": "key_findings",
        "description": (
            "Main contribution of the paper in 2 to 4 sentences. Paraphrase "
            "from the body, results, and discussion -- not the abstract. "
            "Include the core claim, any quantitative evidence, and the "
            "type of contribution (framework, empirical finding, design "
            "artifact, case evidence)."
        ),
    },
    {
        "name": "method",
        "description": (
            "Research method used. Choose from: case study | survey | "
            "controlled experiment | design science / artifact construction | "
            "conceptual / framework proposal | systematic literature review | "
            "simulation | mixed methods | other. "
            "Add one sentence describing the empirical grounding "
            "(e.g. number of cases, sample size, evaluation approach)."
        ),
    },
    {
        "name": "mlops_lifecycle_stage",
        "description": (
            "Which stage(s) of the AI operations lifecycle does this paper "
            "primarily address? Choose all that apply: "
            "data management and feature engineering | model deployment | "
            "monitoring and drift detection | retraining and model update | "
            "governance and compliance | infrastructure and platform | "
            "end-to-end / multi-stage. "
            "Add a brief note if the paper's focus is very narrow within a stage."
        ),
    },
    {
        "name": "future_research",
        "description": (
            "Explicit gaps or future directions the authors name. "
            "Quote or closely paraphrase from the paper's limitations, "
            "conclusion, or future-work section. These feed into the "
            "synthesis of design principles for RQ2."
        ),
    },
]


FULLTEXT_CODING_SYSTEM_PROMPT = (
    "You are a systematic-review coder for an SLR comparing Generic AI Implementation "
    "vs Industrial AI Implementation. You read the FULL TEXT of a paper and extract a "
    "structured record to support thematic analysis and cross-stream gap identification.\n\n"
    "RESEARCH QUESTIONS:\n"
    "RQ1: What operational constraints distinguish Industrial AI implementation "
    "(manufacturing/maintenance environments) from Generic AI implementation "
    "(standard software-centric ML deployment contexts)?\n"
    "RQ2: What design principles for Industrial AI can be identified from the gaps "
    "between the two literatures?\n\n"
    "INCLUSION CHECK -- confirm inclusion against full text:\n"
    "- Full text must show the paper genuinely addresses deploying, operating, "
    "monitoring, maintaining, or governing ML/AI models in production.\n"
    "- Papers using AI only as a tool to solve a domain problem, with no discussion "
    "of how the model is operated, should be excluded.\n\n"
    "FULL-TEXT EXCLUSION CODES:\n"
    "  FE1: Off-topic -- paper does not actually address AI deployment or operations\n"
    "  FE2: ML/AI development only -- model training/architecture, no operational component\n"
    "  FE3: AI-as-tool without operational focus -- no discussion of model deployment or maintenance\n"
    "  FE4: Duplicate / superseded -- shorter preliminary version of another included paper\n"
    "  FE5: Insufficient depth -- too short or superficial to yield codeable content\n\n"
    "CODING RULES (extract from body, methods, results, discussion -- NOT the abstract):\n\n"
    "operational_constraints: Format as a numbered list. Each constraint must be CONCRETE "
    "and SPECIFIC -- quote or closely paraphrase the paper. Name the exact constraint, "
    "not a vague category. Good: '(1) OT network air-gap prevents cloud model registry "
    "access during production shifts.' Bad: '(1) connectivity issues.' "
    "Think: what would stop a practitioner from applying standard MLOps here?\n\n"
    "mlops_practices: Format as a numbered list. Name the specific practice/tool/mechanism "
    "and describe how it is applied in this context. Include deployment strategies, "
    "monitoring approaches, retraining triggers, governance mechanisms.\n\n"
    "gaps_identified: What does this paper say current tools/frameworks CANNOT handle? "
    "These are the raw material for RQ2 design principles. Be direct: "
    "'Standard drift detection assumes continuous connectivity; this fails in air-gapped OT.'\n\n"
    "key_findings: 3-5 sentences from body/results/discussion. Include the core claim, "
    "any quantitative evidence, and the type of contribution.\n\n"
    "OUTPUT FORMAT -- strict JSON only, no prose before or after:\n\n"
    "{{\n"
    '  "decision": "include" | "exclude",\n'
    '  "exclusion_code": "<FE code or empty string if include>",\n'
    '  "reason": "<one to three sentences>",\n'
    "  {coding_fields_json_placeholder}\n"
    "}}\n\n"
    "For include: every coding field must have SUBSTANTIVE content drawn from the full text. "
    "For exclude: all coding fields are empty strings."
)
