"""
Financial Markets Analytics - Final Project Script.
Asset Allocation Robustness and Black-Litterman Framework during the 2026 Conflict.
References: Markowitz (1952), Michaud (1989), Drobetz (2001).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Set global plotting standards
np.random.seed(42)
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_style('whitegrid')

def download_and_clean_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads historical adjusted close prices from Yahoo Finance and resamples to monthly.

    Args:
        tickers (list): List of asset ticker symbols.
        start_date (str): Backtest start date (YYYY-MM-DD).
        end_date (str): Backtest end date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Logarithmic monthly returns matrix.
    """
    print(">>> Downloading asset price series from Yahoo Finance...")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices[tickers].dropna()
    
    # Resample to month-end as recommended for i.i.d. norm properties
    monthly_prices = prices.resample('ME').last()
    log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    print(f">>> Data downloaded successfully. Monthly observations: {len(log_returns)}")
    return log_returns

def get_max_sharpe_weights(mu: np.ndarray, sigma_matrix: np.ndarray, rf_rate: float = 0.0) -> np.ndarray:
    """Computes long-only portfolio weights maximizing the Sharpe Ratio via SLSQP optimization.

    Args:
        mu (np.ndarray): Annualized expected excess returns vector.
        sigma_matrix (np.ndarray): Annualized covariance matrix.
        rf_rate (float): Risk-free rate.

    Returns:
        np.ndarray: Optimized allocation weights summing to 1.
    """
    n_assets = len(mu)
    w0 = np.ones(n_assets) / n_assets
    
    def neg_sharpe(w):
        p_ret = np.dot(w, mu)
        p_vol = np.sqrt(np.dot(w.T, np.dot(sigma_matrix, w)))
        if p_vol == 0: return 0
        return -(p_ret - rf_rate) / p_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-10})
    return res.x if res.success else w0

def get_michaud_weights(mu: np.ndarray, sigma_matrix: np.ndarray, n_sim: int = 50, t_obs: int = 60) -> np.ndarray:
    """Generates robust portfolio weights using Michaud's Resampling methodology.

    Args:
        mu (np.ndarray): Annualized expected returns.
        sigma_matrix (np.ndarray): Annualized covariance matrix.
        n_sim (int): Number of Monte Carlo draws.
        t_obs (int): Length of historical sample size to simulate.

    Returns:
        np.ndarray: Averaged robust portfolio weights.
    """
    n_assets = len(mu)
    mu_m = mu / 12
    cov_m = sigma_matrix / 12
    sim_weights = []

    for _ in range(n_sim):
        draws = np.random.multivariate_normal(mu_m, cov_m, t_obs)
        sim_mu = np.mean(draws, axis=0) * 12
        sim_cov = np.cov(draws.T) * 12
        sim_weights.append(get_max_sharpe_weights(sim_mu, sim_cov))
        
    avg_w = np.mean(sim_weights, axis=0)
    return avg_w / np.sum(avg_w)

def run_black_litterman(mu_hist: np.ndarray, sigma_matrix: np.ndarray, w_eq: np.ndarray, 
                        rf_rate: float, tau: float, P: np.ndarray, Q: np.ndarray) -> tuple:
    """Implements the standard Black-Litterman model framework following Drobetz (2001).

    Args:
        mu_hist (np.ndarray): Historical sample mean returns.
        sigma_matrix (np.ndarray): Sample covariance matrix.
        w_eq (np.ndarray): Equilibrium market capitalization or strategic weights.
        rf_rate (float): Implied risk-free rate.
        tau (float): Scalar weighting factor for prior uncertainty.
        P (np.ndarray): K x N pick matrix mapping views to assets.
        Q (np.ndarray): K x 1 vector of views expressions.

    Returns:
        tuple: Posterior expected returns (mu_BL) and posterior covariance matrix (Sigma_BL).
    """
    inv = np.linalg.inv
    
    # Calibrate risk aversion parameter delta from equilibrium specs
    var_m = np.dot(w_eq.T, np.dot(sigma_matrix, w_eq))
    mu_m = np.dot(w_eq.T, mu_hist)
    delta = (mu_m - rf_rate) / var_m
    
    # Step 1: Reverse Optimization to obtain Implied Equilibrium Returns Pi
    Pi = delta * np.dot(sigma_matrix, w_eq)
    
    # Step 2: Calibrate Omega using the He & Litterman proportional variance convention
    Omega = np.diag(np.diag(np.dot(P, np.dot(tau * sigma_matrix, P.T))))
    
    # Step 3: Compute Posterior Expected Returns (mu_BL) via Master Formula
    term_inv = inv(inv(tau * sigma_matrix) + np.dot(P.T, np.dot(inv(Omega), P)))
    mu_BL = np.dot(term_inv, np.dot(inv(tau * sigma_matrix), Pi) + np.dot(P.T, np.dot(inv(Omega), Q)))
    
    # Step 4: Adjust Posterior Covariance Matrix to incorporate estimation risk
    Sigma_BL = sigma_matrix + term_inv
    
    return Pi, mu_BL, Sigma_BL

def build_efficient_frontier(mu: np.ndarray, sigma_matrix: np.ndarray, n_points: int = 40) -> tuple:
    """Traces out the long-only efficient frontier curve.

    Args:
        mu (np.ndarray): Expected asset returns.
        sigma_matrix (np.ndarray): Covariance matrix.
        n_points (int): Iterative grid density.

    Returns:
        tuple: Realized frontier volatilities, returns, and associated weight vectors.
    """
    n_assets = len(mu)
    w0 = np.ones(n_assets) / n_assets
    
    # Find Minimum Variance Portfolio (MVP) as lower bound
    res_mvp = minimize(lambda w: w @ sigma_matrix @ w, w0, method='SLSQP',
                       bounds=[(0, 1)] * n_assets, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    
    min_ret = res_mvp.x @ mu
    max_ret = mu.max() * 0.99
    
    targets = np.linspace(min_ret, max_ret, n_points)
    f_vols, f_rets, f_weights = [], [], []
    
    for t in targets:
        cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}, {'type': 'eq', 'fun': lambda w: w @ mu - t}]
        res = minimize(lambda w: w @ sigma_matrix @ w, w0, method='SLSQP', bounds=[(0, 1)] * n_assets, constraints=cons)
        if res.success:
            f_vols.append(np.sqrt(res.x @ sigma_matrix @ res.x))
            f_rets.append(res.x @ mu)
            f_weights.append(res.x)
            
    return np.array(f_vols), np.array(f_rets), np.array(f_weights)

def generate_and_save_plots(returns_df: pd.DataFrame, mu: np.ndarray, Sigma: np.ndarray, 
                            w_eq: np.ndarray, Pi: np.ndarray, mu_BL: np.ndarray, Sigma_BL: np.ndarray, 
                            asset_names: list, rf: float):
    """Generates the comprehensive 10-chart analytical visualization set and saves to disk."""
    print(">>> Generating and saving high-resolution analytical chartbook...")
    colors = plt.cm.tab20(np.linspace(0, 1, len(asset_names)))
    
    # Chart 1: Normalized Historical Performance
    plt.figure()
    p_norm = (np.exp(returns_df.cumsum()) * 100)
    p_norm.plot(colormap='tab20', linewidth=1.5)
    plt.title('Asset Class Performance Chartbook (Base 100)')
    plt.ylabel('Normalized Index Value')
    plt.tight_layout()
    plt.savefig('plot/1_etf_performance.png', dpi=300)
    plt.close()
    
    # Chart 2: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(returns_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
    plt.title('Asset Class Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('plot/2_correlation_matrix.png', dpi=300)
    plt.close()
    
    # Chart 3: Reverse Optimization Implied Returns vs Historicals
    plt.figure()
    x = np.arange(len(asset_names))
    plt.bar(x - 0.2, (mu - rf) * 100, 0.4, label='Historical Excess Returns', color='steelblue')
    plt.bar(x + 0.2, Pi * 100, 0.4, label='Implied Equilibrium Returns (Pi)', color='crimson')
    plt.xticks(x, asset_names, rotation=45)
    plt.ylabel('Annual Excess Return (%)')
    plt.title('Reverse Optimization: Historical vs Implied Equilibrium Prior')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/3_reverse_optimization.png', dpi=300)
    plt.close()

    # Chart 4: Black-Litterman Posterior Return Shifts
    plt.figure()
    plt.bar(x - 0.2, Pi * 100, 0.4, label='Prior Equilibrium (Pi)', color='grey', alpha=0.7)
    plt.bar(x + 0.2, mu_BL * 100, 0.4, label='Posterior Distribution (mu_BL)', color='darkblue')
    plt.xticks(x, asset_names, rotation=45)
    plt.ylabel('Annual Excess Return (%)')
    plt.title('Black-Litterman Update: Prior Equilibrium vs View-Adjusted Posterior Mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/4_bl_returns_update.png', dpi=300)
    plt.close()

    # Chart 5: Static Allocation Comparison (Weights Map)
    w_mv = get_max_sharpe_weights(mu - rf, Sigma)
    w_bl_opt = get_max_sharpe_weights(mu_BL, Sigma_BL)
    plt.figure()
    plt.bar(x - 0.3, w_eq * 100, 0.25, label='Equilibrium Baseline Weight', color='grey')
    plt.bar(x, w_mv * 100, 0.25, label='Classical Markowitz Weight', color='firebrick')
    plt.bar(x + 0.3, w_bl_opt * 100, 0.25, label='Black-Litterman Weight', color='teal')
    plt.xticks(x, asset_names, rotation=45)
    plt.ylabel('Portfolio Allocation Weight (%)')
    plt.title('Allocation Comparison: Equilibrium vs Classical vs Black-Litterman')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/5_allocation_comparison.png', dpi=300)
    plt.close()

    # Chart 6: Long-Only Efficient Frontiers Comparison
    v_cls, r_cls, _ = build_efficient_frontier(mu - rf, Sigma)
    v_eq, r_eq, _ = build_efficient_frontier(Pi, Sigma)
    v_bl, r_bl, _ = build_efficient_frontier(mu_BL, Sigma_BL)
    plt.figure()
    plt.plot(v_cls * 100, r_cls * 100, color='firebrick', lw=2.5, label='Classical Markowitz Frontier')
    plt.plot(v_eq * 100, r_eq * 100, color='grey', lw=2, ls='--', label='Equilibrium Prior Frontier')
    plt.plot(v_bl * 100, r_bl * 100, color='teal', lw=2.5, label='Black-Litterman Frontier')
    plt.xlabel('Annualized Portfolio Volatility (%)')
    plt.ylabel('Expected Annual Excess Return (%)')
    plt.title('Efficient Frontiers Comparison Space')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/6_efficient_frontiers.png', dpi=300)
    plt.close()

    # Chart 7: Weight Composition along Classical Markowitz Frontier
    _, _, w_curve_mv = build_efficient_frontier(mu - rf, Sigma)
    plt.figure()
    plt.stackplot(v_cls * 100, w_curve_mv.T * 100, labels=asset_names, colors=colors, alpha=0.85)
    plt.xlim(v_cls.min() * 100, v_cls.max() * 100)
    plt.xlabel('Portfolio Volatility (%)')
    plt.ylabel('Cumulative Asset Allocation Weight (%)')
    plt.title('Weight Map Composition: Classical Markowitz Frontier')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig('plot/7_weight_composition_markowitz.png', dpi=300)
    plt.close()

    # Chart 8: Weight Composition along Black-Litterman Frontier
    _, _, w_curve_bl = build_efficient_frontier(mu_BL, Sigma_BL)
    plt.figure()
    plt.stackplot(v_bl * 100, w_curve_bl.T * 100, labels=asset_names, colors=colors, alpha=0.85)
    plt.xlim(v_bl.min() * 100, v_bl.max() * 100)
    plt.xlabel('Portfolio Volatility (%)')
    plt.ylabel('Cumulative Asset Allocation Weight (%)')
    plt.title('Weight Map Composition: Black-Litterman Frontier')
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig('plot/8_weight_composition_bl.png', dpi=300)
    plt.close()

    # Chart 9: Robustness Check - Perturbed Frontiers Cloud
    plt.figure()
    for _ in range(20):
        perturbed_mu = (mu - rf) + np.random.normal(0, 0.01, len(mu))
        v_p, r_p, _ = build_efficient_frontier(perturbed_mu, Sigma, n_points=20)
        plt.plot(v_p * 100, r_p * 100, color='lightblue', lw=0.8, alpha=0.6)
    plt.plot(v_cls * 100, r_cls * 100, color='firebrick', lw=2.5, label='Baseline Mean-Variance')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Excess Return (%)')
    plt.title('Estimation Error Amplification Cloud (Monte Carlo Mu Perturbations)')
    plt.tight_layout()
    plt.savefig('plot/9_robust_perturbed_frontiers.png', dpi=300)
    plt.close()

def calculate_performance_metrics(strat_returns: pd.DataFrame, rf_rate: float = 0.0) -> pd.DataFrame:
    """Calculates annualized return, volatility, Sharpe Ratio, and Max Drawdown.

    Args:
        strat_returns (pd.DataFrame): Monthly simple returns for each strategy.
        rf_rate (float): Annualized risk-free rate.

    Returns:
        pd.DataFrame: Table of performance metrics.
    """
    metrics = pd.DataFrame(index=strat_returns.columns)
    
    # Annualized Return (compound)
    cum_returns = (1 + strat_returns).prod()
    n_years = len(strat_returns) / 12
    ann_ret = cum_returns ** (1 / n_years) - 1
    metrics['Ann. Return (%)'] = ann_ret * 100
    
    # Annualized Volatility
    ann_vol = strat_returns.std() * np.sqrt(12)
    metrics['Ann. Volatility (%)'] = ann_vol * 100
    
    # Sharpe Ratio
    metrics['Sharpe Ratio'] = (ann_ret - rf_rate) / ann_vol
    
    # Max Drawdown
    wealth_index = (1 + strat_returns).cumprod()
    peak = wealth_index.cummax()
    drawdown = (wealth_index - peak) / peak
    metrics['Max Drawdown (%)'] = drawdown.min() * 100
    
    return metrics.round(2)

def run_rolling_backtest(returns_df: pd.DataFrame, w_eq: np.ndarray, tickers: list, window: int = 60) -> pd.DataFrame:
    """Executes a dynamic out-of-sample backtest with monthly rolling rebalancing windows.

    Args:
        returns_df (pd.DataFrame): Historical asset log returns.
        w_eq (np.ndarray): Predefined strategic equilibrium weights.
        tickers (list): List of asset identifiers.
        window (int): In-sample estimation estimation window length.

    Returns:
        pd.DataFrame: Out-of-sample multi-strategy cumulative returns series.
    """
    oos_index = returns_df.index[window:]
    strat_returns = pd.DataFrame(index=oos_index, columns=['Markowitz', 'Michaud', 'Black-Litterman', 'Equal-Weight', 'SPY Benchmark'])
    
    m_rets, mich_rets, bl_rets, ew_rets, spy_rets = [], [], [], [], []
    
    # Fixed exogenous view specifications parameters for rolling BL setup
    tau = 0.025
    P = np.zeros((2, len(tickers)))
    P[0, tickers.index('USO')] = 1  # Absolute bullish view on Crude Oil
    P[1, tickers.index('ITA')] = 1  # Absolute bullish view on Aerospace/Defense
    
    for i in range(window, len(returns_df)):
        train_slice = returns_df.iloc[i-window:i]
        realized_step = returns_df.iloc[i].values
        current_date = returns_df.index[i]
        
        if current_date.year >= 2025:
            Q = np.array([0.12, 0.10])
        else:
            Q = np.array([0.05, 0.05])
        
        mu_h = train_slice.mean().values * 12
        cov_h = train_slice.cov().values * 12
        
        # 1. Classical Markowitz Weights
        w_mv = get_max_sharpe_weights(mu_h, cov_h)
        # 2. Michaud Resampled Weights
        w_mich = get_michaud_weights(mu_h, cov_h, n_sim=30, t_obs=window)
        # 3. Black-Litterman Weights
        _, mu_bl, cov_bl = run_black_litterman(mu_h, cov_h, w_eq, rf_rate=0.02, tau=tau, P=P, Q=Q)
        w_bl = get_max_sharpe_weights(mu_bl, cov_bl)
        # 4. Naive Benchmark Equal Weights
        w_ew = np.ones(len(tickers)) / len(tickers)
        
        realized_simple = np.exp(realized_step) - 1
        
        m_rets.append(np.dot(w_mv, realized_simple))
        mich_rets.append(np.dot(w_mich, realized_simple))
        bl_rets.append(np.dot(w_bl, realized_simple))
        ew_rets.append(np.dot(w_ew, realized_simple))
        spy_rets.append(realized_simple[tickers.index('SPY')])
        
    strat_returns['Markowitz'] = m_rets
    strat_returns['Michaud'] = mich_rets
    strat_returns['Black-Litterman'] = bl_rets
    strat_returns['Equal-Weight'] = ew_rets
    strat_returns['SPY Benchmark'] = spy_rets
    
    # Chart 10: Backtest Cumulative Wealth Index Line Chart
    plt.figure()
    wealth_index = (1 + strat_returns).cumprod() * 100
    wealth_index.plot(linewidth=2)
    plt.title('Out-of-Sample Wealth Index Tracker Performance')
    plt.ylabel('Portfolio Equity Capital Value')
    plt.axvspan('2025-12-31', '2026-05-31', color='red', alpha=0.1, label='Peak Conflict Crisis Zone')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/10_backtest_wealth_index.png', dpi=300)
    plt.close()
    
    return strat_returns

def main():
    """Main algorithmic pipeline to implement, evaluate and execute the project core."""
    tickers = ['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'LQD', 'GLD', 'USO', 'ITA', 'DBC', 'VNQ', 'XLF', 'XLV', 'XLU', 'SHV']
    
    # 15-year sample lookback data window context
    returns_df = download_and_clean_data(tickers, '2011-01-01', '2026-05-31')
    
    # Calibrate static normalized strategic asset allocation targets vector (w_eq proxy)
    w_eq = np.array([0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.07, 0.05, 0.05, 0.05, 0.04, 0.05, 0.05, 0.05, 0.05])
    w_eq = w_eq / np.sum(w_eq)
    
    # Fixed parameters estimation for full-period cross-sectional analysis
    rf = 0.02
    tau = 0.025
    mu_all = returns_df.mean().values * 12
    Sigma_all = returns_df.cov().values * 12
    
    # Set up subjective view matrices capturing war specific regime projections
    P = np.zeros((2, len(tickers)))
    P[0, tickers.index('USO')] = 1  # Crude Oil absolute bet
    P[1, tickers.index('ITA')] = 1  # Aerospace & Defense sector standalone absolute bet
    Q = np.array([0.12, 0.10])      # Annualized targets projections
    
    # Execute full-sample static Black-Litterman calculations
    Pi, mu_BL, Sigma_BL = run_black_litterman(mu_all, Sigma_all, w_eq, rf, tau, P, Q)
    
    # Compile static chartbook items
    generate_and_save_plots(returns_df, mu_all, Sigma_all, w_eq, Pi, mu_BL, Sigma_BL, tickers, rf)
    
    # Launch out-of-sample dynamic structural rolling backtest
    strat_returns = run_rolling_backtest(returns_df, w_eq, tickers, window=60)
    
    metrics_df = calculate_performance_metrics(strat_returns, rf_rate=0.02)
    print("\n=== OUT-OF-SAMPLE METRICS (2016-2026) ===")
    print(metrics_df.to_string())
    
    strat_returns.to_csv('backtest_results.csv')
    print("\n>>> Results successfully saved to 'backtest_results.csv'.")
    print(">>> Portfolio Optimization analysis complete. Chartbook images successfully outputted to directory.")

if __name__ == '__main__':
    main()