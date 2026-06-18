# Cointegration Analysis: Demographic Ageing and Public Debt

Tests for cointegration between age dependency ratios and structural primary 
balances across 10 developed economies (1960–2023).

## Research Question

Do demographic ageing and fiscal balances share a long-run equilibrium 
relationship? We test whether the two series are cointegrated — i.e. share 
a common stochastic trend despite short-run deviations.

## Methodology

**Step 1 — Unit root tests (by country)**  
ADF and KPSS tests on levels and first differences. Countries classified as 
I(0) or I(1); cointegration analysis restricted to I(1) subsample.

**Step 2 — Engle-Granger two-step procedure**
pb_struct = α + β × Δᵏ(age_dependency) + ε

Residuals tested for stationarity (ADF). Four lag specifications: k = 1, 2, 5, 10 years.  
Multi-period differences capture delayed demographic effects on fiscal stance.

**Step 3 — Cross-country heterogeneity**  
Results interpreted through institutional differences, fiscal rules, and 
demographic transition timing.

## Data

| Variable | Definition | Source |
|----------|-----------|--------|
| `pb_struct` | Structural primary balance = primary balance − 0.5 × output gap | IMF WEO |
| `age_dependency_ratio` | Population 65+ / Population 15–64 | World Bank |

Countries: France, Germany, Italy, Spain, Japan, US, Canada, UK, Netherlands, Sweden.

## Usage

```bash
pip install -r requirements.txt
python cointegration_test.py
```

## References

- Engle & Granger (1987). *Econometrica*, 55(2).
- Dickey & Fuller (1979). *JASA*, 74(366a).
- Kwiatkowski et al. (1992). *Journal of Econometrics*, 54(1–3).

## Author

**Rodrigue Girard**
