# Portfolio Optimization and Risk Metrics Analysis

This project performs portfolio optimization and risk metrics analysis for various investment strategies, focusing on Monte Carlo simulation, risk metrics (VaR, ES), and portfolio weight allocation methods. The code calculates optimal portfolio weights, evaluates portfolio risk metrics, and organizes data to provide insights into performance across different assets under management (AUM) categories.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Functions and Modules](#functions-and-modules)
4. [Usage](#usage)
5. [Requirements](#requirements)

---

### Overview
The primary goal of this project is to apply Monte Carlo simulations and risk metrics calculations on portfolios categorized by different AUM sizes. Each portfolio is analyzed based on its volatility, returns, and other risk measures to identify optimal allocations and risk-adjusted return potential.

### Features
- **Monte Carlo Simulation** for portfolio optimization.
- **Risk Metric Calculations** including Value at Risk (VaR), Expected Shortfall (ES), Sharpe Ratio, and Sortino Ratio.
- **Inverse Volatility Weighting** for optimized portfolio allocation.
- **Portfolio Categorization** into Large, Medium, and Small portfolios for organized performance analysis.

### Functions and Modules

#### 1. **Data Preparation and Portfolio Initialization**
   - `Categorized_Portfolios`: A dictionary categorizing portfolios based on AUM (Large, Medium, Small), and containing portfolios for different strategies (`Normal`, `Growth`, `Value`, and `Diversified`).

#### 2. **Monte Carlo Simulation**
   - `monte_carlo_simulation(returns, cov_matrix, num_simulations=10000)`: Runs Monte Carlo simulations to determine optimal portfolio weights by maximizing the Sharpe ratio.
     - **Parameters**:
       - `returns`: Expected returns for each asset.
       - `cov_matrix`: Covariance matrix for assets.
       - `num_simulations`: Number of simulations to perform.
     - **Returns**: Optimal weights and simulation results (returns, volatility, Sharpe ratio).

#### 3. **Risk Metrics Calculation**
   - `calculate_var_es(daily_returns, weights, confidence_level=0.95)`: Computes VaR and ES at a given confidence level.
   - `calculate_sharpe_ratio(returns, risk_free_rate=0.005)`: Calculates the Sharpe ratio, assuming a specified risk-free rate.
   - `calculate_sortino_ratio(returns, risk_free_rate=0.005)`: Calculates the Sortino ratio, measuring risk-adjusted returns based on downside risk.

#### 4. **Weight Allocation**
   - `inverse_volatility_weights(volatilities)`: Calculates weights inversely proportional to asset volatilities.
   - `calculate_daily_returns_portfolio(portfolio_df)`: Retrieves daily returns for tickers in the given portfolio.

#### 5. **Auxiliary Functions**
   - `get_covariance_matrix(returns_df)`: Computes the covariance matrix for a set of asset returns.
   - `calculate_var_es_fof(returns, confidence_level=0.95)`: Calculates VaR and ES for a Fund of Funds (FoF) portfolio.

### Usage

1. **Setting Up the Portfolios**
   - Define a dictionary of AUM values (Small, Medium, Large) and prepare categorized portfolios using `Categorized_Portfolios`.

2. **Calculating Expected Returns**
   - Use `calculate_daily_returns_portfolio` to get daily returns for each portfolio and store the results.

3. **Running Monte Carlo Simulations**
   - Loop through each portfolio category and run `monte_carlo_simulation` to obtain optimal weights for each portfolio.
   - Store the results in `monte_carlo_results`.

4. **Risk Metrics Calculation**
   - Use `calculate_var_es`, `calculate_sharpe_ratio`, and `calculate_sortino_ratio` to compute risk metrics.
   - Results for each portfolio are stored in `portfolio_risk_metrics` for detailed risk analysis.

5. **Printing Risk Metrics**
   - Risk metrics such as VaR, ES, volatility, Sharpe ratio, and Sortino ratio are printed for each portfolio in each category.

6. **Additional Calculations**
   - The `inverse_volatility_weights` function provides an alternative allocation method based on inverse volatility.

### Requirements
- Python 3.7+
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `yfinance`: To download financial data
