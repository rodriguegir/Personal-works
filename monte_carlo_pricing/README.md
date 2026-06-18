# Monte Carlo Pricer: Multi-Asset Structured Products

Prices a 7-year principal-protected note linked to a basket of 9 US equities 
using correlated Geometric Brownian Motion. Developed as part of quantitative 
finance coursework at Université Paris-Dauphine.

## Product

Annual coupons (years 1–7) based on basket performance, with returns capped 
at +6% and floored at −30%. Principal (100%) returned at maturity regardless 
of performance.

## Methodology

**Path simulation** — Correlated GBM with Cholesky decomposition. Correlation 
matrix corrected for positive semi-definiteness after shocks. NYSE calendar 
(excludes weekends and holidays).

**Discounting** — Zero-coupon curve bootstrapped from short rates, Eurodollar 
futures, and swap rates (2–10y). Cubic spline interpolation.

**Greeks** — Bump-and-revalue method:
- Delta: ±1% move per underlying
- Vega: ±1bp per volatility
- Rho: ±1bp on risk-free rate
- Correlation risk: ±10% shock to correlation matrix

## Usage

> **Recommended: Google Colab** — the notebook includes an interactive 
> dashboard with sliders for all parameters (number of paths, bump sizes, 
> notional, risk-free rate). No local setup required.

Open projet_RG_exotic_options.ipynb in Google Colab

Run Cell 1: defines all functions

Run Cell 2: initializes market data and plots zero-coupon curve

Run Cell 3: launches interactive dashboard

A standalone `cointegration_test.py` is also available for command-line use:
```bash
pip install pandas-market-calendars
python projet_RG_exotic_options.py
```

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

## Author

**Rodrigue Girard**  
MSc Finance and Economics, London School of Economics  
Magistère Banque-Finance-Assurance, Université Paris-Dauphine
