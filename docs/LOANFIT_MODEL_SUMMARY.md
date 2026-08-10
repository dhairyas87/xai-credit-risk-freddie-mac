# LoanFit UI Model Summary

The LoanFit interface exposes four model choices. All four use the same UI
input profile and return the same business outputs: estimated loan amount,
estimated loan term, and estimated interest rate.

| UI name | Version | Model type | What it predicts | Explanation |
|---|---|---|---|---|
| Original mixture | v4 | CatBoost + XGBoost stacked ensemble | Loan amount + loan term + interest rate | This is the advanced research-style loan amount model. CatBoost handles categorical underwriting features directly, XGBoost learns from encoded categorical features, and a Ridge meta-model blends both predictions. The loan term is supplied by the v4 term classifier and interest rate by the v4 CatBoost rate model. |
| Original CatBoost | v4 | CatBoost | Loan amount + loan term + interest rate | This is the direct CatBoost lending model. It is easier to explain than the stacked ensemble because one model produces the loan amount prediction. The loan term is supplied by the v4 term classifier and interest rate by the v4 CatBoost rate model. |
| Stable smart model | v5 | Histogram gradient boosting | Loan amount + loan term + interest rate | This is a UI-safe non-linear comparison model. It captures non-linear relationships while avoiding fragile scikit-learn `ColumnTransformer` pickle artifacts. |
| Stable simple model | v5 | Linear-style baseline | Loan amount + loan term + interest rate | This is the simplest comparison model. It is useful as a baseline because it is faster and easier to explain, but it usually captures fewer non-linear patterns than tree or ensemble models. |

## SHAP explanations

Every UI model option returns SHAP feature drivers for the loan amount estimate:

- Original mixture (v4): SHAP explains the CatBoost base learner inside the
  stacked ensemble.
- Original CatBoost (v4): SHAP explains the direct CatBoost amount model.
- Stable smart model (v5): SHAP explains the histogram gradient boosting amount
  model using the saved background sample in the model bundle.
- Stable simple model (v5): SHAP explains the linear-style amount model using
  the saved background sample in the model bundle.

The UI shows the top drivers, their direction, and their SHAP contribution.

## Why v5 was added

The original v4 models are preserved. The v5 models were added because the UI
hit a scikit-learn serialization issue when loading a saved preprocessing
pipeline across different local environments:

```text
Can't get attribute '_RemainderColsList'
```

The v5 artifacts avoid that specific failure by using a small project-owned
preprocessing bundle in `src/loanfit_modeling.py`.

## Dependency note

The v4 options require the original research dependencies, especially
CatBoost and XGBoost. If those packages are not installed in the active Python
environment, the API will still start, but selecting a v4 model will return a
clear dependency message.
