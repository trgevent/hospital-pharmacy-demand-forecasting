# Hospital Pharmacy Demand Forecasting

**A segment-based forecasting system that turns daily drug consumption into a ready-to-act purchase list — delivered to the pharmacist's inbox every morning.**

> Technical write-up of a project built for *YZM704 – Statistical Machine Learning* (AI Engineering MSc, İzmir Bakırçay University, 2026).
> The project ran on real hospital pharmacy consumption data, which **is not shared** here for confidentiality. This repository documents the architecture, modelling decisions and results.

---

## The problem
A hospital pharmacy manages thousands of drugs with very different demand patterns: some move every day, some a few times a month, some almost never. A single model fails on most of them. Pharmacists also don't need a forecast — they need to know **what to order today, and how much**.

## The approach: route every drug to the model that fits its demand
Drugs are classified by demand density and history, and each segment gets its own method:

| Segment | Demand pattern | Method | Val MAE (boxes/day) |
|---|---|---|---|
| Main – oral | High density | **DNN + LightGBM stacking** | 0.437 |
| Main – parenteral | High volatility | **DNN v6 with Huber loss** | 2.100 |
| Intermittent | 10–60% of days | **TSB (Teunter–Syntetos–Babai)** | 1.174 |
| Low density | < 10% of days | Rolling median | 2.478 |
| Cold start | < 90 active days | ATC-group average | — |
| Obsolete | No movement > 365 days | Monitored, not forecast | — |

The segment of each drug is precomputed into a lookup table, so the API routes requests automatically.

## Architecture
```mermaid
flowchart LR
    A[(SQL Server<br/>stock + daily consumption)] --> B[n8n<br/>daily 08:00 schedule]
    B --> C[Feature engineering<br/>JS code node]
    C -->|HTTP POST JSON<br/>28 features| D[FastAPI model service]
    D --> E{Segment router}
    E --> F[DNN / LightGBM]
    E --> G[TSB]
    E --> H[Rolling median]
    F & G & H --> I[Order decision<br/>+ risk level]
    I --> J[HTML order report<br/>via e-mail]
    D --> K[Streamlit dashboard]
```

**Stack:** n8n (self-hosted, Docker) · FastAPI + uvicorn · TensorFlow/Keras · LightGBM · scikit-learn · SQL Server · Streamlit · SMTP

## Key engineering decisions
- **Box-based normalization.** Consumption is recorded in tablets/ampoules, but orders are placed in boxes. All targets and lag features are divided by pack size, so the model learns in the unit the pharmacist orders in.
- **Cyclic calendar features.** Month and weekday are encoded with sin/cos so December–January and Sunday–Monday are close to each other.
- **No train–serving skew.** The exact scalers fitted during training are persisted and reused at inference.
- **Features computed in the pipeline.** n8n computes lags (1/7/14/30 days) and rolling mean/median/std before calling the API, keeping the model service stateless.
- **Forecast → decision.** `order = 60-day forecast + 15% safety stock − current stock`, plus a risk label:

| Risk | Condition |
|---|---|
| CRITICAL | Stock = 0 |
| WARNING | Stock < safety stock |
| NORMAL | Stock < 2 × safety stock |
| SUFFICIENT | Stock ≥ 2 × safety stock |

## Results

### Model comparison (single-model baseline stage)
| Model | Val MAE | Test MAE |
|---|---|---|
| SARIMA | 4.525 | 5.052 |
| BOCD | 4.459 | 4.854 |
| LSTM | 2.399 | 2.288 |
| DNN | 1.938 | 1.994 |
| XGBoost | 1.916 | 1.957 |
| LightGBM | 1.907 | 1.942 |
| Random Forest | 1.891 | 1.929 |
| **LightGBM Quantile (Q50)** | **1.704** | **1.681** |

Classical time-series models (SARIMA, BOCD) were clearly beaten by feature-based ML. Close scores across ML models with different error patterns motivated stacking; the quantile model's robustness to outliers motivated the Huber loss used later for volatile drugs.

### Parenteral drugs: from 534% to 23% MAPE
| Version | Change | Val MAPE | Val WAPE |
|---|---|---|---|
| v1 | Raw DNN baseline | 534% | — |
| v3 | Log normalization | 49.7% | 49.7% |
| v5 | Per-drug target normalization | 50.4% | 50.4% |
| **v6** | **Dynamic IQR outlier capping + Huber loss (δ=0.5)** | **23.3%** | **14.6%** |

WAPE (14.6%) weights errors by volume, which matters more for pharmacy spend than per-drug MAPE.

### In production
- Runs every morning at 08:00 for the active drug list (~221 drugs), in **45–60 seconds on CPU**
- In a live run, **41 drugs** were automatically flagged with an order quantity and risk level and e-mailed to the pharmacy

## What I would do next
- Prediction intervals instead of a fixed 15% safety stock
- Supplier lead times and minimum order quantities in the order formula
- Drift monitoring and scheduled retraining

## References
- Teunter, Syntetos & Babai (2011). *Intermittent demand forecasting: An empirical comparison.*
- Huber (1964). *Robust Estimation of a Location Parameter.*
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
- Killick, Fearnhead & Eckley (2012). *Optimal Detection of Changepoints with a Linear Computational Cost.*
