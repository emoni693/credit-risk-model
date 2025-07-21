# credit-risk-model


## Credit Scoring Business Understanding

### 1. Basel II Accord and the Need for Interpretability

The Basel II Capital Accord emphasizes risk-sensitive approaches to capital requirements, especially through the Internal Ratings-Based (IRB) approaches. These approaches require financial institutions to measure and manage credit risk rigorously and transparently. As a result, regulatory bodies expect models used for credit scoring to be interpretable, auditable, and well-documented. This ensures that stakeholders—including risk managers, auditors, and regulators—can understand how credit decisions are made, verify model logic, and ensure compliance with regulatory standards. Consequently, models that are transparent (e.g., logistic regression with Weight of Evidence encoding) are often preferred or required, even at the cost of predictive performance.

### 2. The Use of Proxy Variables in the Absence of a "Default" Label

In many real-world datasets, an explicit “default” label may not be available, either due to data limitations or regulatory constraints. In such cases, a proxy variable—such as "late payments beyond 90 days" or "charged-off status"—is used to simulate default behavior. However, using a proxy introduces assumptions and business risks. If the proxy does not accurately represent true default behavior, model predictions may be biased or misaligned with actual risk. This misalignment could lead to poor lending decisions, financial losses, and compliance issues if regulators deem the proxy inadequate or misleading.

### 3. Trade-Offs: Simple vs. Complex Models in Regulated Environments

In credit risk modeling, especially under regulatory oversight, there's a critical trade-off between interpretability and performance. Simple models like Logistic Regression with Weight of Evidence (WoE) offer transparency, stability, and ease of explanation, making them suitable for environments that demand accountability and documentation. However, these models may underperform in capturing complex patterns in data.

On the other hand, advanced models like Gradient Boosting Machines (GBM) offer superior accuracy by capturing non-linear relationships and interactions. Yet, they are often considered "black-box" models, making them harder to explain and validate. In regulated contexts, the preference often leans toward simpler models unless the complex models are accompanied by strong interpretability tools (e.g., SHAP values), rigorous validation, and regulatory approval.
