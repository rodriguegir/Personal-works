"""
Cointegration Analysis: Demographic Aging and Fiscal Policy
Author: Rodrigue Girard
Description: Multi-country analysis of cointegration between age dependency
             ratios and structural primary balances.

Methodology:
    Part 1: Exploratory unit root tests across 10 developed economies
    Part 2: Cointegration analysis on I(1) subsample
    Part 3: Comparative discussion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import PooledOLS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def test_stationarity(series, name, test_type="both", display=True):
    """
    Test stationarity using ADF and/or KPSS tests.

    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Variable name for display
    test_type : str
        "adf", "kpss", or "both"
    display : bool
        Whether to print results

    Returns:
    --------
    dict : Test statistics and p-values
    """
    results = {}

    if test_type in ["adf", "both"]:
        adf_result = adfuller(series.dropna())
        results['adf_stat'] = adf_result[0]
        results['adf_pval'] = adf_result[1]
        results['adf_stationary'] = adf_result[1] < 0.05

        if display:
            print(f"\n--- {name} (ADF Test) ---")
            print(f"ADF statistic: {adf_result[0]:.4f}")
            print(f"p-value: {adf_result[1]:.4f}")
            print("=> Stationary (I(0))" if adf_result[1] < 0.05 else "=> Non-stationary (likely I(1))")

    if test_type in ["kpss", "both"]:
        kpss_stat, kpss_pval, _, kpss_crit = kpss(series.dropna(), regression='c', nlags="auto")
        results['kpss_stat'] = kpss_stat
        results['kpss_pval'] = kpss_pval
        results['kpss_stationary'] = kpss_pval > 0.05

        if display:
            print(f"\n--- {name} (KPSS Test) ---")
            print(f"KPSS statistic: {kpss_stat:.4f}")
            print(f"p-value: {kpss_pval:.4f}")
            print("=> Stationary (I(0))" if kpss_pval > 0.05 else "=> Non-stationary")

    return results


def create_differenced_vars(df, group_col="country"):
    """
    Create first and second differences of key variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataset
    group_col : str
        Column to group by (default: "country")

    Returns:
    --------
    pd.DataFrame : DataFrame with added difference variables
    """
    df = df.copy()

    # first differences
    df["dpb"] = df.groupby(group_col)["pb_struct"].diff()
    df["ddebt"] = df.groupby(group_col)["debt_to_gdp"].diff()
    df["dage"] = df.groupby(group_col)["age_dependency_ratio"].diff()

    # second differences
    df["ddpb"] = df.groupby(group_col)["dpb"].diff()
    df["dddebt"] = df.groupby(group_col)["ddebt"].diff()
    df["ddage"] = df.groupby(group_col)["dage"].diff()

    # multi-period differences
    df["age_diff_2"] = df.groupby(group_col)["age_dependency_ratio"].diff(2)
    df["age_diff_5"] = df.groupby(group_col)["age_dependency_ratio"].diff(5)
    df["age_diff_10"] = df.groupby(group_col)["age_dependency_ratio"].diff(10)

    return df


def engle_granger_test(y, x, name="", display=True):
    """
    Perform Engle-Granger cointegration test.

    Parameters:
    -----------
    y : pd.Series
        Dependent variable
    x : pd.Series
        Independent variable
    name : str
        Test description
    display : bool
        Whether to print results

    Returns:
    --------
    dict : Test results
    """
    stat, pval, _ = coint(y, x)

    if display:
        print(f"\n--- Engle-Granger Test: {name} ---")
        print(f"Test statistic: {stat:.4f}")
        print(f"p-value: {pval:.4f}")
        print("=> Cointegrated" if pval < 0.05 else "=> Not cointegrated")

    return {'stat': stat, 'pval': pval, 'cointegrated': pval < 0.05}


def pooled_ols_cointegration(df, y_var, x_vars, test_name=""):
    """
    Run pooled OLS and test residuals for stationarity.

    Parameters:
    -----------
    df : pd.DataFrame
        Panel data with MultiIndex (country, year)
    y_var : str
        Dependent variable name
    x_vars : list
        List of independent variable names
    test_name : str
        Description of the test

    Returns:
    --------
    dict : Regression results and cointegration test
    """
    # ensure proper indexing
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (country, year)")

    # add constant if not present
    X = df[x_vars].copy()
    if 'const' not in x_vars:
        X = sm.add_constant(X)

    y = df[y_var]

    # estimate model
    model = PooledOLS(y, X)
    results = model.fit(cov_type="robust")

    print(f"\n{'='*70}")
    print(f"Pooled OLS: {test_name}")
    print(f"{'='*70}")
    print(results.summary)

    # test residuals for stationarity
    residuals = results.resids.dropna()
    adf_result = adfuller(residuals, regression="c")

    print(f"\nResidual stationarity test (ADF):")
    print(f"  Test statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print("  => Cointegrated (residuals are I(0))" if adf_result[1] < 0.05
          else "  => Not cointegrated (residuals are I(1))")

    return {
        'model': results,
        'adf_pval': adf_result[1],
        'cointegrated': adf_result[1] < 0.05
    }


def plot_series_by_country(df, variables, countries, title=""):
    """
    Plot time series by country for visual inspection.

    Parameters:
    -----------
    df : pd.DataFrame
        Panel dataset with 'country' and 'year' columns
    variables : list
        Variables to plot
    countries : list
        Countries to include
    title : str
        Plot title
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
    if n_vars == 1:
        axes = [axes]

    for idx, var in enumerate(variables):
        for country in countries:
            subset = df[df["country"] == country]
            if len(subset) > 0 and var in subset.columns:
                axes[idx].plot(subset["year"], subset[var], label=country, alpha=0.7)

        axes[idx].set_title(f"{var} - {title}")
        axes[idx].set_xlabel("year")
        axes[idx].set_ylabel(var)
        axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 1: EXPLORATORY UNIT ROOT TESTS
# ============================================================================

def part1_exploratory_tests(df):
    """
    Part 1: Test all countries to identify integration properties.

    Goal: Determine which countries have I(0) vs I(1) primary balances,
          as cointegration framework only applies when variables are I(1).
    """

    print("\n" + "="*70)
    print("PART 1: EXPLORATORY UNIT ROOT TESTS ACROSS COUNTRIES")
    print("="*70)
    print("\nObjective: Identify integration order of key variables by country")
    print("This determines which subsample is appropriate for cointegration analysis.")

    # define all countries in dataset
    all_countries = ["France", "Germany", "Italy", "Spain", "Japan",
                    "United States", "Canada", "United Kingdom",
                    "Netherlands", "Sweden"]

    df_all = df[df["country"].isin(all_countries)].copy()
    df_all = create_differenced_vars(df_all)

    print(f"\nCountries tested: {', '.join(all_countries)}")
    print(f"Sample period: {df_all['year'].min()} - {df_all['year'].max()}")

    # visual inspection of levels
    print("\n--- Visual inspection: levels ---")
    plot_series_by_country(
        df_all,
        ["pb_struct", "age_dependency_ratio"],
        all_countries,
        "levels"
    )

    # test each country individually
    print("\n" + "-"*70)
    print("INDIVIDUAL COUNTRY TESTS: PRIMARY BALANCE (pb_struct)")
    print("-"*70)

    country_results = {}

    for country in all_countries:
        df_country = df_all[df_all["country"] == country].copy()

        print(f"\n{'='*70}")
        print(f"COUNTRY: {country}")
        print(f"{'='*70}")

        # test levels
        print("\n[LEVELS]")
        pb_level = test_stationarity(df_country["pb_struct"],
                                     f"{country} - pb_struct",
                                     test_type="both")

        age_level = test_stationarity(df_country["age_dependency_ratio"],
                                      f"{country} - age_dependency_ratio",
                                      test_type="adf")

        # test first differences
        print("\n[FIRST DIFFERENCES]")
        df_country_diff = df_country.dropna(subset=["dpb", "dage"])

        pb_diff = test_stationarity(df_country_diff["dpb"],
                                   f"{country} - Δ(pb_struct)",
                                   test_type="both")

        age_diff = test_stationarity(df_country_diff["dage"],
                                     f"{country} - Δ(age)",
                                     test_type="adf")

        # classification
        pb_is_i1 = (not pb_level['adf_stationary']) and pb_diff['adf_stationary']
        pb_is_i0 = pb_level['adf_stationary']

        country_results[country] = {
            'pb_level_pval': pb_level['adf_pval'],
            'pb_diff_pval': pb_diff['adf_pval'],
            'classification': 'I(1)' if pb_is_i1 else ('I(0)' if pb_is_i0 else 'Ambiguous'),
            'suitable_for_cointegration': pb_is_i1
        }

    # summary table
    print("\n" + "="*70)
    print("SUMMARY: INTEGRATION ORDER BY COUNTRY")
    print("="*70)

    summary_df = pd.DataFrame(country_results).T
    summary_df = summary_df.sort_values('classification')
    print("\n", summary_df.to_string())

    # identify subsamples
    i1_countries = [c for c, r in country_results.items()
                   if r['suitable_for_cointegration']]
    i0_countries = [c for c, r in country_results.items()
                   if r['classification'] == 'I(0)']

    print(f"\n{'='*70}")
    print("COUNTRY GROUPINGS FOR FURTHER ANALYSIS")
    print(f"{'='*70}")
    print(f"\nI(1) countries (suitable for cointegration): {', '.join(i1_countries)}")
    print(f"I(0) countries (level relationship): {', '.join(i0_countries)}")

    print("\nInterpretation:")
    print("- I(1) countries: Primary balance is non-stationary in levels but")
    print("  stationary in first differences. Cointegration framework applies.")
    print("- I(0) countries: Primary balance is stationary in levels.")
    print("  Standard regression appropriate (no cointegration needed).")

    return i1_countries, i0_countries, df_all


# ============================================================================
# PART 2: COINTEGRATION ANALYSIS (I(1) SUBSAMPLE)
# ============================================================================

def part2_cointegration_analysis(df, i1_countries):
    """
    Part 2: Cointegration testing on countries where framework applies.

    Focus: France, United States, United Kingdom (I(1) subsample)
    """

    print("\n" + "="*70)
    print("PART 2: COINTEGRATION ANALYSIS (I(1) SUBSAMPLE)")
    print("="*70)

    print(f"\nSubsample: {', '.join(i1_countries)}")
    print("These countries have I(1) primary balances, making them suitable")
    print("for Engle-Granger cointegration testing.")

    # filter to I(1) countries
    df_i1 = df[df["country"].isin(i1_countries)].copy()
    df_i1 = create_differenced_vars(df_i1)
    df_i1 = df_i1.dropna()

    print(f"\nObservations: {len(df_i1)}")
    print(f"Time span: {df_i1['year'].min()} - {df_i1['year'].max()}")

    # visual check of differenced variables
    print("\n--- Visual stationarity check: first differences ---")
    plot_series_by_country(
        df_i1,
        ["dpb", "dage"],
        i1_countries,
        "first differences (should be stationary)"
    )

    # formal stationarity tests on pooled sample
    print("\n" + "-"*70)
    print("STATIONARITY VERIFICATION (POOLED SAMPLE)")
    print("-"*70)

    print("\n[LEVELS - should be I(1)]")
    test_stationarity(df_i1["pb_struct"], "pb_struct (pooled)")
    test_stationarity(df_i1["age_dependency_ratio"], "age_dependency_ratio (pooled)")

    print("\n[FIRST DIFFERENCES - should be I(0)]")
    test_stationarity(df_i1["dpb"], "Δ(pb_struct) (pooled)")
    test_stationarity(df_i1["dage"], "Δ(age) (pooled)")

    # engle-granger cointegration tests
    print("\n" + "-"*70)
    print("ENGLE-GRANGER COINTEGRATION TESTS")
    print("-"*70)

    print("\nRationale: Testing different age specifications to capture the")
    print("slow-moving nature of demographic transitions.")

    eg_results = {}

    # test 1: first difference
    eg_results['dage'] = engle_granger_test(
        df_i1["pb_struct"],
        df_i1["dage"],
        "pb_struct ~ Δ(age)"
    )

    # test 2: 5-year difference
    eg_results['age_diff_5'] = engle_granger_test(
        df_i1["pb_struct"],
        df_i1["age_diff_5"],
        "pb_struct ~ Δ5(age)"
    )

    # test 3: 10-year difference
    eg_results['age_diff_10'] = engle_granger_test(
        df_i1["pb_struct"],
        df_i1["age_diff_10"],
        "pb_struct ~ Δ10(age)"
    )

    # pooled ols estimations with residual tests
    print("\n" + "-"*70)
    print("POOLED OLS REGRESSIONS WITH RESIDUAL STATIONARITY TESTS")
    print("-"*70)

    # prepare panel index
    df_panel = df_i1.copy()
    df_panel = df_panel.set_index(["country", "year"])

    # model specifications
    models = {
        "Model 1: Δ(age)": ["dage"],
        "Model 2: Δ2(age)": ["age_diff_2"],
        "Model 3: Δ5(age)": ["age_diff_5"],
        "Model 4: Δ10(age)": ["age_diff_10"]
    }

    ols_results = {}

    for model_name, x_vars in models.items():
        result = pooled_ols_cointegration(
            df_panel,
            "pb_struct",
            x_vars,
            model_name
        )

        coef_name = x_vars[0]
        ols_results[model_name] = {
            'coefficient': result['model'].params[coef_name],
            'std_error': result['model'].std_errors[coef_name],
            'p_value': result['model'].pvalues[coef_name],
            'r_squared': result['model'].rsquared,
            'n_obs': result['model'].nobs,
            'adf_pval': result['adf_pval'],
            'cointegrated': result['cointegrated']
        }

    # comparative results
    print("\n" + "="*70)
    print("COMPARATIVE RESULTS: COINTEGRATION MODELS")
    print("="*70)

    results_df = pd.DataFrame(ols_results).T
    print("\n", results_df.to_string())

    print("\nInterpretation Guide:")
    print("- Coefficient: Long-run impact of age variable on primary balance")
    print("- p_value < 0.05: Statistically significant relationship")
    print("- adf_pval < 0.05: Residuals are stationary → cointegration confirmed")
    print("- r_squared: Proportion of variance explained")

    # diagnostic tests
    print("\n" + "-"*70)
    print("DIAGNOSTIC TESTS")
    print("-"*70)

    # breusch-pagan on model 1
    X_bp = df_panel[["dage"]]
    X_bp = sm.add_constant(X_bp)
    y_bp = df_panel["pb_struct"]

    bp_model = PooledOLS(y_bp, X_bp)
    bp_res = bp_model.fit()

    _, bp_pval, _, _ = het_breuschpagan(
        bp_res.resids,
        bp_res.model.exog.dataframe.values
    )

    print(f"\nBreusch-Pagan test (Model 1): p-value = {bp_pval:.4f}")
    if bp_pval < 0.05:
        print("=> Heteroskedasticity detected")
        print("   Implication: Robust standard errors are appropriate (already used)")
    else:
        print("=> No significant heteroskedasticity")

    # residual plot
    plt.figure(figsize=(10, 5))
    plt.scatter(bp_res.fitted_values, bp_res.resids, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("residuals vs fitted values - model 1: pb_struct ~ Δ(age)")
    plt.xlabel("fitted values")
    plt.ylabel("residuals")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ols_results


# ============================================================================
# PART 3: COMPARATIVE DISCUSSION
# ============================================================================

def part3_discussion(i1_countries, i0_countries):
    """
    Part 3: Discuss findings and cross-country heterogeneity.
    """

    print("\n" + "="*70)
    print("PART 3: DISCUSSION AND INTERPRETATION")
    print("="*70)

    print("\n1. CROSS-COUNTRY HETEROGENEITY")
    print("-" * 70)

    print(f"\nI(1) countries: {', '.join(i1_countries)}")
    print("Characteristics:")
    print("- Primary balances exhibit persistent shocks (unit root)")
    print("- Fiscal adjustments are slow and path-dependent")
    print("- Suitable for cointegration analysis with demographic variables")

    print(f"\nI(0) countries: {', '.join(i0_countries)}")
    print("Characteristics:")
    print("- Primary balances are mean-reverting")
    print("- Fiscal adjustments are faster (automatic stabilizers or fiscal rules)")
    print("- Standard regression more appropriate than cointegration")

    print("\n2. POTENTIAL EXPLANATIONS FOR HETEROGENEITY")
    print("-" * 70)

    print("\nInstitutional factors:")
    print("- Fiscal rules: Countries with binding fiscal rules (e.g., Germany's")
    print("  debt brake) may have stationary fiscal balances")
    print("- Political systems: Parliamentary vs. presidential systems may")
    print("  exhibit different fiscal dynamics")
    print("- EU membership: Stability and Growth Pact constraints")

    print("\nEconomic structure:")
    print("- Automatic stabilizers: Size of welfare state affects cyclical")
    print("  adjustment speed")
    print("- Debt levels: High-debt countries may face market discipline")
    print("- Demographic transition timing: Countries at different stages")

    print("\n3. IMPLICATIONS FOR COINTEGRATION RESULTS")
    print("-" * 70)


    print("\n No cointegration:")
    print("- No stable long-run relationship")
    print("- Aging effects may be offset by other factors (productivity, pension reforms)")
    print("- Alternative specifications needed (structural breaks, nonlinearity)")

    print("\n4. LIMITATIONS AND EXTENSIONS")
    print("-" * 70)

    print("\nCurrent limitations:")
    print("- Small sample size ")
    print("- Assumes homogeneous coefficients across countries")
    print("- Does not account for structural breaks (financial crisis, COVID); for more precision we could have added dummies")

    print("\nPossible extensions:")
    print("- Country-specific Vector Error Correction Models (VECM)")
    print("- Regime-switching models for structural breaks")
    print("- Include additional controls (debt levels, growth, interest rates)")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main analysis workflow:
    Part 1: Exploratory tests across all countries
    Part 2: Cointegration analysis on I(1) subsample
    Part 3: Discussion and interpretation
    """

    print("="*70)
    print("COINTEGRATION ANALYSIS: DEMOGRAPHIC AGING AND FISCAL POLICY")
    print("="*70)
    print("\nResearch Question:")
    print("Do structural primary balances and age dependency ratios exhibit")
    print("a long-run equilibrium relationship in developed economies?")

    # load data
    print("\n" + "-"*70)
    print("DATA LOADING")
    print("-"*70)

    df = pd.read_csv("data/final_dataset_ageing_test.csv")

    # create structural primary balance
    df["pb_struct"] = df["gvt_primary_balance"] - 0.5 * df["output_gap"]
    df["const"] = 1

    # filter years
    df = df[df["year"] >= 1960].copy()

    print(f"\nDataset loaded: {len(df)} observations")
    print(f"Time period: {df['year'].min()} - {df['year'].max()}")
    print(f"Countries: {df['country'].nunique()}")

    # part 1: exploratory tests
    i1_countries, i0_countries, df_all = part1_exploratory_tests(df)

    # part 2: cointegration analysis
    if len(i1_countries) > 0:
        ols_results = part2_cointegration_analysis(df, i1_countries)
    else:
        print("\nWarning: No I(1) countries identified. Cointegration analysis skipped.")
        ols_results = {}

    # part 3: discussion
    part3_discussion(i1_countries, i0_countries)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return {
        'i1_countries': i1_countries,
        'i0_countries': i0_countries,
        'cointegration_results': ols_results
    }


if __name__ == "__main__":
    results = main()
