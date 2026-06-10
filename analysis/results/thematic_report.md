# Thematic Analysis Report — MLOps / Industrial AI SLR

**Included papers:** 180 (134 generic, 46 industrial)

---

## RQ1: Operational Constraints

### Generic AI Implementation — Themes

#### TG1: Data Dynamics & Quality Management
*Technical constraint — 80 papers*

This theme addresses the inherent challenges of managing data in AI systems, including its continuous evolution, quality issues, and the need for robust processes to ensure data integrity and relevance throughout the ML lifecycle. It highlights how data's dynamic nature directly impacts model performance and operational stability.

**Sub-themes:** Data drift and concept drift, Data scarcity and imbalance, Data quality and validation, Data versioning and provenance

#### TG2: Resource Constraints & Performance Optimization
*Technical constraint — 75 papers*

This theme focuses on the limitations of computational resources (CPU, GPU, memory, power) across various deployment environments, from tiny edge devices to cloud infrastructure. It emphasizes the need for model optimization, efficient resource allocation, and scalable architectures to meet performance, latency, and cost requirements.

**Sub-themes:** Limited computational power/memory, Real-time latency requirements, Energy consumption and cost, Scalability and dynamic resource allocation

#### TG3: MLOps Process Automation & Standardization
*Technical constraint — 90 papers*

This theme covers the need for automating and standardizing the entire ML lifecycle, from experimentation to deployment and monitoring. It highlights the challenges of manual processes, lack of consistent methodologies, and the demand for integrated tools and frameworks to streamline MLOps workflows.

**Sub-themes:** End-to-end pipeline automation, CI/CD/CT integration, Experiment tracking and reproducibility, Standardization of practices and tools

#### TG4: Trustworthy AI & Governance
*Mixed constraint — 65 papers*

This theme addresses the critical need for ethical, transparent, and compliant AI systems, particularly in high-stakes domains. It encompasses challenges related to bias, explainability, privacy, security, and regulatory adherence, emphasizing the integration of governance throughout the ML lifecycle.

**Sub-themes:** Ethical AI and bias mitigation, Explainable AI (XAI), Privacy and data security, Regulatory compliance and auditability

#### TG5: Deployment Complexity & Environment Heterogeneity
*Technical constraint — 70 papers*

This theme explores the difficulties of deploying and managing AI models across diverse and often distributed environments, including cloud, edge, on-premises, and hybrid setups. It highlights challenges related to system integration, interoperability, and adapting models to varying operational contexts.

**Sub-themes:** Hybrid/multi-cloud deployments, Edge/TinyML integration, System integration and interoperability, Containerization and microservices

#### TG6: Organizational & Collaboration Gaps
*Organisational constraint — 50 papers*

This theme addresses the human and organizational challenges in implementing MLOps, including skill gaps, lack of cross-functional collaboration, and the disconnect between technical and business stakeholders. It emphasizes the need for clear roles, shared understanding, and integrated workflows to bridge these divides.

**Sub-themes:** Skill gaps and specialized expertise, Cross-functional collaboration, Business value alignment, Organizational maturity

#### TG7: Model Monitoring & Adaptive Retraining
*Technical constraint — 60 papers*

This theme focuses on the necessity of continuously monitoring deployed models for performance degradation, drift, and anomalies, and implementing automated or semi-automated retraining strategies. It highlights the challenges of real-time detection, label latency, and ensuring model integrity over time.

**Sub-themes:** Continuous model performance monitoring, Drift detection (data, concept, label), Automated/adaptive retraining, Label latency and ground truth availability


### Industrial AI Implementation — Themes

#### TI1: Edge-Cloud Continuum for Industrial AI
*Technical constraint — 20 papers*

Industrial AI deployments frequently span from resource-constrained edge devices to powerful cloud infrastructure. This continuum is essential for balancing real-time processing needs, data residency, and computational demands, but introduces complexity in deployment and management.

**Sub-themes:** Resource-constrained edge devices, Low-latency requirements, Hybrid deployment strategies, Distributed data processing

#### TI2: Data Dynamics and Quality Challenges
*Technical constraint — 25 papers*

Industrial environments are characterized by dynamic, often messy, and scarce data. This includes issues like data drift, limited labeled data, and data quality degradation, which severely impact model performance and necessitate robust data management and adaptation strategies.

**Sub-themes:** Concept and data drift, Data scarcity and imbalance, Data quality and messiness, Real-time data processing

#### TI3: Integration with Legacy and OT Systems
*Technical constraint — 15 papers*

A significant structural challenge in industrial AI is the need to integrate new ML solutions with existing operational technology (OT) and legacy IT systems. This involves dealing with diverse protocols, outdated hardware, and established infrastructure, often leading to complex integration efforts.

**Sub-themes:** Heterogeneous OT layer, Legacy hardware and software, Air-gapped networks, PLC integration

#### TI4: Security, Privacy, and Regulatory Compliance
*Mixed constraint — 12 papers*

Industrial AI deployments operate under strict security, data privacy, and regulatory requirements. This often restricts the use of public cloud solutions, necessitates on-premises deployments, and demands robust identity and access management, especially for sensitive industrial or health data.

**Sub-themes:** Data confidentiality and privacy, Regulatory constraints (e.g., FDA), Network isolation and air-gapped environments, Trust and reliability of ML applications

#### TI5: Automated and Adaptive ML Lifecycle Management
*Technical constraint — 22 papers*

The dynamic nature of industrial environments necessitates highly automated and adaptive MLOps practices. This includes continuous monitoring, automated drift detection, and self-healing or retraining mechanisms to maintain model accuracy and operational stability over time.

**Sub-themes:** Continuous monitoring and drift detection, Automated retraining and redeployment, Self-healing ML systems, Feedback loops for model adaptation

#### TI6: Human-in-the-Loop and Domain Expertise
*Organisational constraint — 10 papers*

Despite automation, human involvement and domain expertise remain crucial in industrial AI. This includes iterative collaboration between domain experts and ML engineers, the need for explainable AI (XAI) to build trust, and the challenge of integrating human feedback into automated pipelines.

**Sub-themes:** Collaboration between domain experts and ML engineers, Explainable AI (XAI) for trust and understanding, Human feedback integration, Sparse ML knowledge in SMEs

#### TI7: Lack of Standardized Industrial MLOps Frameworks
*Technical constraint — 18 papers*

There is a significant gap in standardized, comprehensive MLOps platforms and methodologies specifically tailored for industrial applications. Existing tools often focus on cloud-native, generic ML, or lack the specific functionalities needed for the unique constraints of industrial environments.

**Sub-themes:** Absence of native industrial MLOps platforms, Limited tools for cyber-physicality of IoT, Focus on cloud-centric vs. industrial needs, Lack of integrated data governance

---

## Cross-Stream Gap Analysis

| Industrial Theme | Present in Generic? | Gap Level | Why Generic Falls Short |
|-----------------|--------------------|-----------|-----------------------|
| Edge-Cloud Continuum for Industrial AI | partially | **MEDIUM** | Generic MLOps often treats edge as a distinct deployment target, whereas industrial AI demands seamless, often bi-direct |
| Data Dynamics and Quality Challenges | True | **LOW** | Generic MLOps practices for data management, drift detection, and quality assurance are largely transferable. While indu |
| Integration with Legacy and OT Systems | False | **HIGH** | Generic MLOps frameworks are not designed to handle the complexities of integrating with heterogeneous, often proprietar |
| Security, Privacy, and Regulatory Compliance | True | **MEDIUM** | Generic MLOps often focuses on data privacy and ethical AI in a broader enterprise context. Industrial AI's regulatory l |
| Automated and Adaptive ML Lifecycle Management | True | **LOW** | Generic MLOps practices for monitoring and adaptive retraining are largely applicable. While industrial environments mig |
| Human-in-the-Loop and Domain Expertise | True | **MEDIUM** | Generic MLOps often focuses on bridging gaps between data scientists and business stakeholders. Industrial AI requires a |
| Lack of Standardized Industrial MLOps Frameworks | False | **HIGH** | Generic MLOps frameworks are typically designed for cloud-native or enterprise software deployments and lack the specifi |

---

## RQ2: Design Principles for Industrial AI Implementation

### DP-1: Integrate with Legacy and Operational Technology (OT) Systems
*Evidence strength: strong*

**Principle:** Industrial AI systems must incorporate specialized interfaces, protocols, and security measures to seamlessly integrate with existing legacy and operational technology (OT) infrastructure, including PLCs, SCADA, and air-gapped networks.

**Rationale:** Generic MLOps frameworks are not designed to handle the complexities of integrating with heterogeneous, often proprietary, and outdated industrial control systems. This principle addresses the critical need for native support for industrial communication protocols and hardware interfaces to enable AI deployment in real-world industrial settings.

**Implementation guidance:**
- Develop or adopt specialized connectors and protocol converters for common industrial communication standards (e.g., Modbus, OPC UA, PROFINET).
- Implement robust data acquisition strategies that account for data formats, sampling rates, and reliability of legacy OT systems.
- Design secure, one-way data diodes or air-gapped solutions for transferring data from critical OT networks to IT/cloud environments for AI processing.
- Establish clear data governance and access control policies that respect the security and operational integrity of OT systems.

### DP-2: Establish Native Industrial MLOps Frameworks
*Evidence strength: strong*

**Principle:** Industrial AI implementations require the development or adoption of MLOps frameworks specifically designed to manage the lifecycle of AI models within cyber-physical systems, real-time OT integration, and highly distributed edge-cloud architectures.

**Rationale:** Generic MLOps frameworks lack the specific functionalities and architectural considerations required for industrial cyber-physical systems and legacy OT. This principle emphasizes the need for tailored platforms that can automate ML functions, manage model lifecycle, and integrate with unique industrial constraints beyond general-purpose tools.

**Implementation guidance:**
- Prioritize MLOps platforms that offer native support for industrial data sources, real-time processing, and deployment to resource-constrained edge devices.
- Develop automated pipelines for data ingestion, model training, validation, and deployment that are resilient to industrial network variability and data quality issues.
- Implement robust model versioning and rollback mechanisms that can operate within the constraints of industrial control systems.
- Integrate capabilities for continuous monitoring of model performance and data drift specifically tailored for industrial sensor data and operational contexts.

### DP-3: Orchestrate Edge-Cloud Continuum for Industrial AI
*Evidence strength: moderate*

**Principle:** Industrial AI systems must implement sophisticated orchestration and data flow management across a tightly integrated, often real-time, and highly distributed edge-cloud continuum, accounting for heterogeneous operational environments and resource constraints.

**Rationale:** Generic MLOps often treats edge as a distinct deployment target, whereas industrial AI demands seamless, latency-critical interaction across the entire edge-cloud spectrum. This principle addresses the need for holistic management of AI workloads, data processing, and model updates across deeply integrated, heterogeneous operational environments.

**Implementation guidance:**
- Design a distributed architecture that intelligently partitions AI workloads between edge devices and cloud infrastructure based on latency, bandwidth, and computational requirements.
- Implement robust data synchronization and communication protocols optimized for intermittent connectivity and varying network conditions in industrial settings.
- Develop mechanisms for remote model deployment, updates, and monitoring on resource-constrained edge devices, including over-the-air (OTA) updates.
- Utilize federated learning or similar decentralized approaches to train models on local edge data while preserving data privacy and minimizing data transfer to the cloud.

### DP-4: Ensure Industrial-Grade Security, Privacy, and Regulatory Compliance
*Evidence strength: moderate*

**Principle:** Industrial AI systems must embed security, privacy, and regulatory compliance measures that meet stringent, often safety-critical, industry-specific standards, including requirements for on-premises deployments, air-gapped networks, and 'locked' models.

**Rationale:** Industrial AI's regulatory landscape is often tied to physical safety, critical infrastructure, and specific industry standards, leading to more rigid constraints on model modification, data residency, and network architecture than generic MLOps anticipates. This principle ensures AI systems adhere to these heightened requirements.

**Implementation guidance:**
- Implement robust access control and authentication mechanisms tailored for OT environments, including multi-factor authentication and role-based access.
- Design data encryption strategies for data at rest and in transit that comply with industry-specific security standards and data residency requirements.
- Establish immutable model deployment and versioning processes for safety-critical applications, preventing unauthorized modifications and ensuring auditability.
- Conduct regular security audits and penetration testing specifically targeting the unique vulnerabilities of industrial control systems and AI integrations.

### DP-5: Integrate Human-in-the-Loop with Domain Expertise
*Evidence strength: moderate*

**Principle:** Industrial AI systems must deeply integrate human domain experts (e.g., process engineers, maintenance specialists) into the AI lifecycle, providing mechanisms for continuous feedback, operational validation, and interpretation of model decisions in safety-critical contexts.

**Rationale:** Industrial AI requires a much deeper, continuous integration with highly specialized operational domain experts, where model explainability and human feedback are essential for operational trust and safety, not just business understanding. This principle ensures AI systems are validated and trusted by those with critical operational knowledge.

**Implementation guidance:**
- Design user interfaces and dashboards that provide explainable AI (XAI) insights tailored for domain experts, highlighting key features influencing model predictions.
- Establish formal feedback loops where operational personnel can validate model outputs, flag anomalies, and provide input for model retraining and refinement.
- Develop simulation environments or digital twins that allow domain experts to test and validate AI model behavior under various operational scenarios before real-world deployment.
- Provide training and collaboration platforms that bridge the knowledge gap between data scientists and industrial domain experts, fostering mutual understanding and trust.
