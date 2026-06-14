# Key Analysis and Research Findings

## Research Finding 1: Credit Score is the Strongest Serviceability Indicator

Analysis revealed a clear relationship between borrower credit score and future financial stress.

Average Credit Score:

| Stress Flag | Average Credit Score |
| ----------- | -------------------- |
| Healthy     | 749.53               |
| Stressed    | 722.36               |

Observation:

* Stressed borrowers originate with substantially lower credit scores.
* Credit score consistently decreases as BSSI severity increases.

Conclusion:

Credit Score emerged as the strongest predictor of future borrower stress and was therefore assigned the highest weight within the Borrower Serviceability Score (BSS).

---

## Research Finding 2: Debt-to-Income Ratio Influences Borrower Stress

Average DTI:

| Stress Flag | Average DTI |
| ----------- | ----------- |
| Healthy     | 35.32       |
| Stressed    | 37.97       |

Observation:

* Stressed borrowers exhibit higher debt burdens.
* Borrowers with greater debt obligations appear more vulnerable to future financial stress.

Conclusion:

DTI was selected as a core component of the proposed Serviceability framework.

---

## Research Finding 3: Interest Rate Impacts Serviceability

Average Interest Rate:

| Stress Flag | Average Interest Rate |
| ----------- | --------------------- |
| Healthy     | 4.37%                 |
| Stressed    | 4.49%                 |

Observation:

* Borrowers experiencing stress generally originated with higher interest rates.
* Higher interest rates increase monthly repayment obligations.

Conclusion:

Interest Rate contributes meaningfully to borrower serviceability and was incorporated into the BSS framework.

---

## Research Finding 4: Multiple Borrowers Improve Financial Resilience

Distribution of Borrowers:

Healthy Loans:

* Single Borrower: 52.3%
* Multiple Borrowers: 47.7%

Stressed Loans:

* Single Borrower: 63.1%
* Multiple Borrowers: 36.9%

Observation:

* Financial stress is more common among single-borrower loans.
* Multi-borrower loans demonstrate greater resilience.

Conclusion:

Income diversification through multiple borrowers appears to improve mortgage serviceability.

---

## Research Finding 5: LTV Contributes to Stress but is a Secondary Driver

Average LTV:

| Stress Flag | Average LTV |
| ----------- | ----------- |
| Healthy     | 74.43       |
| Stressed    | 75.56       |

Observation:

* Higher leverage is associated with increased borrower stress.
* The relationship is weaker than Credit Score, DTI, and Interest Rate.

Conclusion:

LTV was retained as a supporting feature within the Serviceability framework.

---

## Research Finding 6: Proposed BSSI Successfully Captures Stress Severity

Borrower Serviceability Stress Index (BSSI) was introduced as a custom stress severity measure.

Distribution:

| BSSI | Percentage |
| ---- | ---------- |
| 0    | 86.22%     |
| 1    | 10.26%     |
| 2    | 3.46%      |
| 3    | 0.057%     |

Observation:

* Stress severity decreases naturally across the portfolio.
* Severe stress events are rare but identifiable.

Conclusion:

BSSI provides an interpretable borrower stress hierarchy suitable for portfolio segmentation and severity analysis.

---

## Research Finding 7: Proposed BSS Demonstrates Strong Predictive Separation

Borrowers were segmented into Serviceability Buckets.

Stress Rates:

| BSS Bucket | Stress Rate |
| ---------- | ----------- |
| Very Low   | 25.93%      |
| Low        | 17.16%      |
| Medium     | 13.03%      |
| High       | 9.29%       |
| Very High  | 5.70%       |

Observation:

* Stress rates decline monotonically as serviceability improves.
* Borrowers with poor serviceability are significantly more likely to experience future stress.

Conclusion:

The proposed Borrower Serviceability Score effectively captures differences in borrower risk.

---

## Research Finding 8: Very Low Serviceability Borrowers are 4.5x More Likely to Experience Stress

Risk Ratio:

25.93% / 5.70% = 4.55

Observation:

Borrowers within the Very Low Serviceability category are approximately 4.5 times more likely to experience future financial stress than borrowers within the Very High Serviceability category.

Conclusion:

The BSS framework provides meaningful and actionable borrower segmentation.

---

## Research Finding 9: Financial Stress Remains Stable Across Quarters

Stress Rates:

| Quarter | Stress Rate |
| ------- | ----------- |
| 2018Q1  | 14.41%      |
| 2018Q2  | 13.61%      |
| 2018Q3  | 13.74%      |
| 2018Q4  | 13.39%      |

Observation:

Stress prevalence remains remarkably stable throughout 2018.

Conclusion:

The dataset exhibits limited temporal drift and is suitable for temporal validation experiments.

---

## Research Finding 10: Serviceability Scores Remain Stable Across Time

Average BSS:

| Quarter | Average BSS |
| ------- | ----------- |
| 2018Q1  | 52.32       |
| 2018Q2  | 52.13       |
| 2018Q3  | 51.80       |
| 2018Q4  | 49.29       |

Observation:

Serviceability remains stable across most quarters, with a modest decline observed during Q4.

Conclusion:

Temporal information should be preserved during modeling to capture potential shifts in borrower quality.

---

# Primary Research Contribution

This project introduces two novel borrower-centric constructs:

1. Borrower Serviceability Stress Index (BSSI)

   * Measures borrower stress severity.

2. Borrower Serviceability Score (BSS)

   * Measures borrower repayment capacity at origination.

Together, these frameworks provide an interpretable bridge between traditional mortgage origination characteristics and future borrower financial outcomes.

The upcoming modeling phase will evaluate whether the proposed BSS framework improves predictive performance relative to traditional credit risk models.

