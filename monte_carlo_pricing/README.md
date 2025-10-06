Monte Carlo Pricer for Multi-Asset Structured Products
Overview
This project implements a Monte Carlo simulation framework to price and analyze exotic structured notes. The code was developed as part of my quantitative finance coursework at Université Paris Dauphine and demonstrates proficiency in numerical methods, derivatives pricing, and risk management.
Product Description
The pricer values a 7-year principal-protected note linked to a basket of 9 US equities with the following payoff structure:
Annual Coupons (Years 1-7):

Observation date: April 25th each year
For each asset: calculate return vs. initial price
Apply performance adjustments:

Cap positive returns at +6%
Floor negative returns at -30%


Coupon = max(0, arithmetic average of adjusted returns) × nominal × time factor
Coupon paid only if basket performance is positive

Maturity (Year 7):

Principal (100% of nominal) returned regardless of performance
Final coupon paid if applicable

Methodology
1. Path Simulation

Model: Correlated Geometric Brownian Motion
Correlation handling: Cholesky decomposition of correlation matrix with positive semi-definite correction
Parameters: Risk-free rate, dividend yield, asset volatilities
Calendar: NYSE trading days (excludes weekends and holidays)

2. Discounting

Zero-coupon curve bootstrapped from market data:

Short rates (overnight to 6 months)
Eurodollar futures (medium-term rates)
Swap rates (2-10 years)


Cubic spline interpolation with outlier detection
All cashflows discounted to valuation date

3. Risk Analytics
Greeks calculated via bump-and-revalue method:

Delta: Price sensitivity to ±1% move in each underlying
Vega: Price sensitivity to ±1bp move in each volatility
Rho: Price sensitivity to ±1bp move in risk-free rate
Correlation risk: Sensitivity to ±10% shock to correlation matrix

4. Numerical Stability

Correlation matrix correction ensures positive semi-definiteness after shocks
Fixed random seed for reproducibility within simulations
Vectorized operations for computational efficiency

Code Structure
projet_RG_exotic_options.ipynb
├── Cell 1: Core functions
│   ├── nearest_positive_definite()      # Matrix correction
│   ├── generate_paths()                 # Monte Carlo simulation
│   ├── construct_zero_coupon_curve()    # Curve bootstrapping
│   ├── calculate_payoffs()              # Product payoff logic
│   ├── calculate_product_price()        # Price aggregation
│   ├── calculate_*_sensitivities()      # Greeks computation
│   └── create_dashboard()               # Interactive interface
├── Cell 2: Parameter initialization
│   └── Market data and product specs
└── Cell 3: Dashboard execution
    └── Interactive widget interface
Dependencies
pythonnumpy
pandas
scipy
matplotlib
pandas_market_calendars
ipywidgets
Install with:
bashpip install pandas-market-calendars
Usage
PyCharm or Google Colab(Recommended) 

For Google Colab usage:
Upload projet_RG_exotic_options.ipynb to Google Colab
Run Cell 1 (install dependencies + define functions)
Run Cell 2 (initialize parameters + plot zero-coupon curve)
Run Cell 3 (launch interactive dashboard)
Adjust parameters via sliders and click "Run Simulation"

Parameters (adjustable via dashboard)

Asset prices and volatilities: Initial market conditions
Number of simulations: Monte Carlo paths (100-100,000)
Nominal: Product notional amount
Risk-free rate: Discounting and drift adjustment
Bump sizes: Custom shocks for sensitivity analysis

Delta bump (%)
Vega bump (bps)
Rho bump (bps)
Correlation bump (bps)



Output
The dashboard displays:

Product Price: Fair value and % of nominal
Simulation Parameters: Inputs used in valuation
Greeks: Aggregate sensitivities (Delta, Vega, Rho, Corr)
Risks Summary: Impact of specified bumps on price
Delta/Vega by Asset: Individual contributions to portfolio Greeks
Performance Table: Detailed path-by-path analysis with coupons and discounted payoffs

Key Features

Correlation matrix validation: Automatic correction to ensure positive semi-definiteness
Realistic calendar: Uses actual NYSE trading days
Market data integration: Bootstraps curve from observable rates
Interactive analysis: Real-time parameter adjustments without re-running code
Reproducibility: Fixed random seed per simulation

Limitations and Extensions
Current Limitations

Assumes constant volatility (no stochastic vol)
Lognormal asset dynamics (no jumps)
Constant correlation structure
No early redemption features (autocall)

Possible Extensions

Implement variance reduction techniques (antithetic variates, control variates)
Add Heston stochastic volatility model
Incorporate jump-diffusion dynamics
Extend to autocallable structures
Add Monte Carlo Greeks (pathwise derivatives)

Academic Context
This project demonstrates:

Numerical methods for derivative pricing (Monte Carlo simulation)
Linear algebra for correlation structure (Cholesky decomposition, eigenvalue analysis)
Financial mathematics (risk-neutral pricing, discounting)
Software engineering (modular code, interactive visualization)

These skills are directly applicable to quantitative research in asset pricing, risk management, and computational finance.
References

Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
Higham, N. J. (2002). Computing the nearest correlation matrix. IMA Journal of Numerical Analysis, 22(3), 329-343.
Hull, J. C. (2018). Options, Futures, and Other Derivatives (10th ed.). Pearson.

Author
Rodrigue Girard
MSc Finance and Economics, London School of Economics
Magistère in Banking, Finance, and Insurance, Université Paris Dauphine-PSL
