Monte Carlo Pricer for Exotic Structured Product
Overview
This code implements a Monte Carlo simulation framework to price and analyze a structured note with the following features:

Multi-asset basket (9 US equities)
Annual coupons based on capped/floored basket performance
7-year maturity with principal protection at maturity

Key Components
1. Path Generation (generate_paths)

Simulates correlated Geometric Brownian Motion paths for all assets
Uses Cholesky decomposition to incorporate correlation structure
Accounts for risk-free rate and dividend yield

2. Payoff Structure (calculate_payoffs)

Annual observation dates (April 25th each year)
For each observation:

Calculate each asset's performance vs. initial price
Apply cap (+6% max) and floor (-30% min) to individual performances
Coupon = max(0, arithmetic mean of adjusted performances)


Principal returned at maturity
All cashflows discounted using zero-coupon curve

3. Zero-Coupon Curve (construct_zero_coupon_curve)

Bootstraps curve from market data (overnight rates, futures, swap rates)
Applies outlier detection and cubic spline interpolation
Used for discounting future cashflows

4. Greeks Calculation
Implements bump-and-revalue for:

Delta: sensitivity to each underlying price (1% bump)
Vega: sensitivity to each volatility (1bp bump)
Rho: sensitivity to interest rates (1bp bump)
Corr: sensitivity to correlation matrix (10% bump)

Uses nearest_positive_definite to ensure correlation matrices remain valid after shocks.
5. Interactive Dashboard (create_dashboard)

Jupyter widget interface to adjust all parameters
Real-time recalculation on "Run Simulation" button
Displays: product price, Greeks, individual sensitivities, performance table

Technical Notes

Uses fixed seed for reproducibility within each simulation
Handles correlation matrix corrections to maintain positive semi-definiteness
Vectorized operations for computational efficiency
Trading calendar based on NYSE schedule (excludes weekends/holidays)

Typical Use Case
Price a worst-of autocallable note, analyze sensitivities to market parameters, and stress-test under different scenarios (volatility spikes, correlation breakdowns, rate moves).
