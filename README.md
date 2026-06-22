# Financial Market Analytics

## Dynamic Asset Allocation under Geopolitical Stress
### A Comparative Study of Markowitz, Michaud Resampling, and Black–Litterman Portfolios

This project investigates dynamic asset allocation strategies in a diversified ETF universe from 2010 to 2026, with a particular focus on portfolio robustness during periods of geopolitical uncertainty.

The analysis compares three portfolio construction frameworks:

- **Markowitz Mean-Variance Optimization**
- **Michaud Resampled Efficiency**
- **Black–Litterman Model with Geopolitical Views**

The main research objective is to evaluate which framework provides the most robust risk-adjusted performance during market stress, specifically during the 2026 Iran/Hormuz geopolitical crisis scenario.

---

## Repository Structure

```text
.
├── financial_market_analytics_unified.ipynb
├── data/
├── figures/
├── results/
├── README.md
└── report.pdf
```

---

## Research Question

> Which portfolio optimization framework provides the most robust risk-adjusted allocation during market stress, especially during the Iran/Hormuz geopolitical episode of 2026?

The project adopts a rolling out-of-sample backtesting framework where portfolio weights are re-estimated monthly using a 60-month historical window.

---

## ETF Universe

The investment universe includes a broad set of liquid ETFs representing:

- Developed Equities
- Emerging Market Equities
- Government Bonds
- Corporate Credit
- Real Estate
- Commodities
- Precious Metals
- Energy
- Inflation-Protected Securities
- Cash Equivalents

Example ETFs:

| Asset Class | ETFs |
|------------|-------|
| US Equities | SPY, QQQ |
| International Equities | VGK, EWJ, VPL |
| Emerging Markets | EEM, VWO |
| Bonds | SHY, IEF, TLT, IGOV, EMB |
| Credit | LQD, HYG |
| Commodities | DBC, USO |
| Precious Metals | GLD, SLV |
| Energy | XLE |
| Real Estate | VNQ |
| Cash Proxy | BIL |

---

## Methodology

### 1. Markowitz Optimization

Implementation of classical mean-variance portfolio optimization including:

- Global Minimum Variance Portfolio
- Target Return Minimum Variance Portfolio
- Maximum Sharpe Ratio Portfolio
- Maximum Utility Portfolios

---

### 2. Michaud Resampling

To mitigate estimation error, the project implements Michaud's Resampled Efficiency methodology:

- Monte Carlo simulations
- Resampled efficient frontiers
- Averaged portfolio weights
- Improved allocation stability

---

### 3. Black–Litterman Model

The Black–Litterman framework combines:

- Market equilibrium returns
- Bayesian updating
- Subjective geopolitical views

The model incorporates tactical views related to the Strait of Hormuz crisis, including:

- Oil supply shocks
- Safe-haven demand for gold
- Energy sector outperformance
- Relative regional equity performance

---

## Key Features

- Dynamic monthly rebalancing
- Rolling-window estimation
- Efficient frontier construction
- Portfolio backtesting
- Risk-adjusted performance evaluation
- Drawdown analysis
- Correlation regime analysis
- Geopolitical stress testing

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- yFinance
- Plotly
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aronetommaso/Financial-Market-Analytics.git
cd Financial-Market-Analytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
financial_market_analytics_unified.ipynb
```

---

## Results

The empirical analysis shows that:

- Classical Markowitz portfolios can achieve strong returns but are highly sensitive to estimation error and often become concentrated.
- Michaud Resampling improves diversification and allocation stability.
- Black–Litterman achieves the best risk-adjusted performance during geopolitical stress by combining market equilibrium information with economically grounded tactical views.

---

## Reproducibility

All analyses, figures, portfolio optimizations, and backtests can be reproduced directly from the notebook provided in this repository.

---

## Authors

**Tommaso Arone**  
Matricola: **896282**

**Lorenzo Triolo**  
Matricola: **895541**

---

## Academic Information

**Course:** Financial Market Analytics  
**Academic Year:** 2025/2026

---

## License

This project is intended for academic and educational purposes.
