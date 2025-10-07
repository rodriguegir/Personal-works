# Fiscal Reaction Functions and Debt Sustainability Limits

Replication and extension of Ghosh et al. (2013) fiscal reaction function and debt limit analysis for 23 OECD countries (1980-2022)

## Overview

This project estimates maximum sustainable debt-to-GDP ratios by analyzing how primary fiscal balance responds to debt levels. The methodology follows Ghosh et al. (2013) using cubic regression models to capture potential "fiscal fatigue" effects.

## Data Sources

All data retrieved from IMF DataMapper API:
- **Debt**: Government gross debt (% GDP)
- **Primary Balance**: General government primary net lending/borrowing (% GDP)
- **Interest Payments**: Interest paid on public debt (% GDP)
- **GDP Growth**: Real GDP growth rate
- **Inflation**: Average consumer price inflation
- **Government Expenditure**: Primary government expenditure (% GDP)

## Methodology

### Fiscal Reaction Function

**Two model specifications estimated:**

**1. Static model (replicating Ghosh et al. with AR(1), and without including govt_exp_gap):**

pb_t = μ + β₁·debt_{t-1} + β₂·debt²_{t-1} + β₃·debt³_{t-1} + β₄·output_gap + ε_t

**2. Dynamic model (with AR correction and adding a lag):**

pb_t = μ + β₁·debt_{t-1} + β₂·debt²_{t-1} + β₃·debt³_{t-1} + β₄·output_gap + β₅·pb_{t-1} + ε_t

- Both estimated using **pggls** (panel generalized GLS) with country fixed effects
- Dynamic specification includes lagged dependent variable to correct autocorrelation
- Output gap computed via HP filter (λ=100)

### r-g Differential

Historical 10-year rolling average of:
- **r** = implied nominal interest rate (interest payments / lagged debt)
- **g** = nominal GDP growth (real growth + inflation)

### Debt Limit (d̄)

Deterministic debt limit = maximum intersection between:
1. **Fiscal reaction function** (steady-state for dynamic model)
2. **Interest payment schedule**: (r* - g) × d

For dynamic model, steady-state reaction = f(d) / (1 - β₅), assuming output gap = 0 in long run.

## Key Results

The models estimate debt limits (or not if doesn't exist) for all countries:
- First model maintains methodological consistency with Ghosh et al.
- Dynamic model addresses econometric concerns 
- Countries with negative r-g differentials show higher sustainable debt levels
- Cubic specification captures non-linear fiscal responses

## Files
- `gosh_et_al_R_v2.R`: Main analysis script
- `gosh_et_al_R_v2.ipynb`: Main analysis script for Colab

## Usage

### Google Colab (R environment)
1. Open Google Colab and select R runtime (Select "Execution" in the upper bar and "modify execution type"-> there select R)
2. Upload the script file
3. Run all cells sequentially

### RStudio
1. Open `gosh_et_al_R_v2.R` in RStudio
2. Install required packages (see below)
3. Run the script
4. Note: You may need to adjust file paths for local environment

## Requirements
```r
install.packages(c("httr", "jsonlite", "dplyr", "tidyr", "plm", 
                   "lmtest", "sandwich", "mFilter", "zoo"))


Model Choice
We present both specifications:

Static model for methodological replication of Ghosh et al.
Dynamic model for more econometric robustness

The static model prioritizes comparability with original research (but presence of autocorrelation), while the dynamic model addresses residual autocorrelation present in fiscal data.

Caveats
This is a very simplistic replication 
Econometric robustness is weak
But the purpose is more to use econometric tools

References
Ghosh, A. R., Kim, J. I., Mendoza, E. G., Ostry, J. D., & Qureshi, M. S. (2013). Fiscal fatigue, fiscal space and debt sustainability in advanced economies. The Economic Journal, 123(566), F4-F30.

