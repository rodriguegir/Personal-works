Cointegration Analysis: Demographic Aging and Fiscal Policy
Overview
This project tests for cointegration between age dependency ratios and 
structural primary balances across developed economies using the 
Engle-Granger methodology. The analysis follows a three-stage approach: 
(1) exploratory unit root tests to identify integration properties by 
country, (2) cointegration testing on the appropriate subsample, and (3) 
interpretation of cross-country heterogeneity.
Research Question
Do structural primary balances and demographic aging exhibit a long-run 
equilibrium relationship in developed economies? Specifically, we test 
whether fiscal balances and age dependency ratios are cointegrated—sharing 
a common trend despite potential short-run deviations.

Methodology
Part 1: Exploratory Unit Root Tests
Objective: Determine which countries have I(0) versus I(1) primary 
balances.
Why this matters: Cointegration tests only apply when variables are 
integrated of the same order (typically I(1)). Testing each country 
individually allows us to:

Identify cross-country heterogeneity in fiscal dynamics
Select the appropriate econometric framework by country
Understand institutional and structural differences

Tests applied:

Augmented Dickey-Fuller (ADF): Tests null hypothesis of unit root 
(non-stationarity)
KPSS: Tests null hypothesis of stationarity
Applied to both levels and first differences

Country classification:

I(1) countries: Primary balance non-stationary in levels, stationary in 
first differences → suitable for cointegration
I(0) countries: Primary balance stationary in levels → cointegration not relevant

Part 2: Cointegration Analysis
Sample: Countries where both variables are I(1) (typically France, United 
States, United Kingdom, Italy, Japan based on preliminary tests)
Engle-Granger Two-Step Procedure:

Estimate cointegrating regression: pb_struct = α + β * age_variable + ε
Test residuals for stationarity using ADF
If residuals are I(0), variables are cointegrated

Specifications tested:

Model 1: pb_struct ~ Δ(age) - annual first difference
Model 2: pb_struct ~ Δ2(age) - 2-year difference
Model 3: pb_struct ~ Δ5(age) - 5-year difference
Model 4: pb_struct ~ Δ10(age) - 10-year difference

Rationale: Multi-period differences may better capture impact of demographic changes on future primary balance. That is demographic effects of public finances can occur with delay.

Part 3: Discussion
Interprets findings in light of:

Institutional differences (fiscal rules, political systems)
Economic structure (automatic stabilizers, debt levels)
Demographic transition timing
Policy implications of cointegration results

Data
Source: IMF, World Bank

Variables:

pb_struct: Structural primary balance (% of potential GDP)

Calculated as: primary_balance - 0.5 × output_gap
The choice of 0.5 was motivated by OECD findings of this value for developed countries.
Cyclically-adjusted measure of discretionary fiscal stance

age_dependency_ratio: (Population 65+ / Population 15-64) × 100

debt_to_gdp: General government gross debt (% of GDP)

Countries analyzed:
France, Germany, Italy, Spain, Japan, United States, Canada, United 
Kingdom, Netherlands, Sweden
Sample period: 1960-2023 (subject to data availability)
Frequency: Annual

Installation
Requirements
bashpip install -r requirements.txt
requirements.txt:
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
statsmodels>=0.14.0
linearmodels>=4.27
openpyxl>=3.0.0

Project Structure
cointegration_analysis/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── cointegration_test.py              # Main analysis script
├── data/
│   └── final_dataset_ageing_test.csv  # Your dataset (Excel converted 
to CSV)
└── output/                            # (optional) For saving plots

Usage
Step 1: Prepare your data

Required columns in your dataset:

country: Country name (string)
year: Year (integer)
gvt_primary_balance: Primary balance (% of GDP)
output_gap: Output gap (% of potential GDP)
age_dependency_ratio: Age dependency ratio
debt_to_gdp: Debt-to-GDP ratio

Step 2: Run the analysis
Option A: Command line (recommended)
bashpython cointegration_test.py

Option B: Google Colab

Upload files to Colab:

Upload cointegration_test.py
Create a data/ folder and upload your CSV
Change the path to read the civ file.

Install dependencies:

python!pip install linearmodels

Run:

python%run cointegration_test.py

Option C: Jupyter Notebook
python# In a notebook cell
%run cointegration_test.py

Step 3: Interpret the output

The script produces three main sections:
PART 1 Output:

Individual country unit root tests
Summary table of integration orders
Country groupings (I(0) vs I(1))
Time series plots for visual inspection

PART 2 Output:

Stationarity tests on pooled I(1) sample
Engle-Granger test statistics
Pooled OLS regression results
Comparative table of model specifications
Diagnostic tests (Breusch-Pagan for heteroskedasticity)
Residual plots

PART 3 Output:

Interpretation of cross-country heterogeneity
Discussion of institutional factors
Policy implications
Limitations and extensions

Expected Results
Interpretation Guide
If cointegration is found (residual ADF p-value < 0.05):

✓ Long-run equilibrium relationship exists
✓ Coefficient β represents persistent fiscal impact of aging
✓ Deviations from equilibrium trigger corrective adjustments
✓ Policy implication: Demographic trends constrain fiscal space

If no cointegration (residual ADF p-value > 0.05):

✗ No stable long-run relationship
✗ Variables may be correlated but relationship isn't persistent
→ Consider alternative specifications (structural breaks, nonlinearity)

Key Output Tables

Country Classification Table

Shows which countries are I(0) vs I(1)
Justifies subsample selection


Comparative Model Results

Coefficient, standard error, p-value, R²
Residual stationarity test (ADF p-value)
Cointegration verdict (yes/no)


Diagnostic Tests

Heteroskedasticity (Breusch-Pagan test)
Residual plots for visual inspection



Troubleshooting
Common Issues
Error: "File not found"

Ensure CSV is in data/ folder
Check filename exactly matches: final_dataset_ageing_test.csv

Error: "KeyError: column not found"

Verify your CSV has required columns (see Step 1)
Check column names have no extra spaces

Warning: "No I(1) countries identified"

Check your data quality (sufficient observations per country)
Verify time series are long enough (typically need 30+ years)
May indicate all countries have stationary balances (less common)


Running on Different Datasets
To adapt this code for your own data:

Modify variable names in main():

pythondf["pb_struct"] = df["your_primary_balance_column"] - 0.5 * 
df["your_output_gap_column"]

Change country list:

pythonall_countries = ["Country1", "Country2", ...]  # in 
part1_exploratory_tests()

Adjust time filter:

pythondf = df[df["year"] >= YOUR_START_YEAR].copy()
Key Econometric Concepts
I(0) (Stationary):

Mean-reverting series
Shocks have temporary effects
Standard regression appropriate

I(1) (Unit root):

Non-stationary in levels
Shocks have permanent effects
First difference is stationary
Requires cointegration framework

Cointegration:

Two I(1) series share common stochastic trend
Linear combination is I(0)
Implies long-run equilibrium relationship
Short-run deviations are corrected over time

Spurious Regression Problem:

Regressing unrelated I(1) variables produces misleading results
High R² without meaningful relationship
Cointegration tests protect against this

Extensions for Further Research

Panel cointegration tests: Pedroni (1999), Kao (1999) for larger samples
Vector Error Correction Model (VECM): Captures short-run dynamics and 
adjustment speed
Structural break tests: Gregory-Hansen (1996) for regime changes
Nonlinear cointegration: Threshold models for asymmetric adjustment
Additional controls: Debt levels (Bohn type analysis), interest rates, growth, political 
variables

References

Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error 
correction: Representation, estimation, and testing. Econometrica, 55(2), 
251-276.
Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for 
autoregressive time series with a unit root. Journal of the American 
Statistical Association, 74(366a), 427-431.
Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). Testing 
the null hypothesis of stationarity against the alternative of a unit 
root. Journal of Econometrics, 54(1-3), 159-178.
Phillips, P. C., & Perron, P. (1988). Testing for a unit root in time 
series regression. Biometrika, 75(2), 335-346.

Author
Rodrigue Girard
MSc Finance and Economics, London School of Economics
Supervised by Dr. Anton Brender, Université Paris Dauphine-PSL

Quick Start Example
python# After installing dependencies and preparing data:
python cointegration_test.py

# Output will show:
# Part 1: Which countries are I(0) vs I(1)
# Part 2: Cointegration tests and regressions for I(1) countries
# Part 3: Discussion of findings
