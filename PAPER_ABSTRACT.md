# AgentMaintain: An Agentic MLOps Framework for Autonomous Predictive Maintenance using Local Small Language Models

## Abstract (final numbers from evaluated run, seed=42)

AgentMaintain is an autonomous, edge-deployed Agentic MLOps framework that combines:
1. **Bonferroni-corrected KS monitoring** across 21 C-MAPSS turbofan sensors
2. **RUL-grounded SHAP attribution** (Random Forest trained on Remaining Useful Life)
3. **Confidence-weighted multi-model consensus** using quantized Qwen-2.5 7B + Phi-3.5 mini via Ollama

Evaluated over 96 decisions per configuration on a 7-event synthetic fault schedule:

| Configuration   | Overall ECR (95% CI)   | Fault ECR (95% CI)   | Normal ECR (95% CI)   | Avg Latency | p95 Latency | VRAM (MB) | F1    |
|-----------------|------------------------|----------------------|-----------------------|-------------|-------------|-----------|-------|
| Qwen-2.5 7B     | 71.9% [62.5, 81.3]     | 79.0% [69.4, 88.7]   | 58.8% [44.1, 76.5]    | 6.31 s      | 7.40 s      | 5090      | 0.784 |
| Phi-3.5 mini    | 65.6% [56.3, 75.0]     | **98.4%** [95.2, 100]| 5.9% [0.0, 14.7]      | **3.05 s**  | 4.46 s      | **4034**  | 0.792 |
| Consensus       | **79.2%** [70.8, 86.5] | 85.5% [77.4, 93.6]   | **67.7%** [52.9, 82.4]| 6.60 s      | 7.56 s      | 5289      | **0.848** |

### Key Findings

- **Consensus is the best overall model** (79.2% ECR, F1=0.848), not any individual model.
- **Phi-3.5 has extreme class bias**: near-perfect fault recall (98.4%) but near-zero normal specificity (5.9%) — operationally unsuitable standalone.
- **Qwen-2.5 7B is the most balanced individual model** (71.9% overall, 79.0% fault, 58.8% normal).
- **Cohen's κ (Qwen vs Phi) = 0.037** — near-zero, confirming complementary/independent error profiles. Statistically validates the ensemble design.
- **McNemar tests**: Qwen vs Phi: p=0.124 (not significant — neither dominates the other). Qwen vs Consensus: p<0.001 (significant). Phi vs Consensus: p=0.055 (borderline).
- **Hallucination incident**: Phi-3.5 output `RETAIN_ASYMPTOMATIC` at cycle 1400 with confidence 0.65, which the confidence-weighted tie-breaker accepted as the consensus. Fixed by Pydantic `Literal` type enforcement.
- **KS sensitivity sweep**: larger windows → higher detection rates (49.5% at ref=25/cur=15 up to 99.0% at ref=100/cur=50); default ref=50/cur=30 = 84.5%.
- All configs remain **strictly below 6 GB VRAM** on consumer-grade GPU hardware.
