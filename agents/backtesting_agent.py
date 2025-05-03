#%% Importing libraries and data

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load your strategy data
df = pd.read_excel('backtest.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Download benchmark data
start_date = df.index[0].strftime('%Y-%m-%d')
end_date = df.index[-1].strftime('%Y-%m-%d')

# Get NASDAQ (^IXIC) and SPX (^GSPC) data
benchmarks = yf.download(['^IXIC', '^GSPC'], start=start_date, end=end_date, interval='1mo')['Close']
benchmarks = benchmarks.resample('M').last()  # Ensure monthly frequency matches your data
benchmarks = benchmarks.pct_change().dropna()

#%% Strategy metrics calculation

# Calculate strategy metrics
initial_capital = 100  # Assuming $100 initial capital
strategy_cumulative = (1 + df['str_return']).cumprod() * initial_capital
strategy_total_return = (strategy_cumulative[-1] - initial_capital) / initial_capital * 100

# Calculate benchmark metrics
nasdaq_cumulative = (1 + benchmarks['^IXIC']).cumprod() * initial_capital
nasdaq_total_return = (nasdaq_cumulative[-1] - initial_capital) / initial_capital * 100

spx_cumulative = (1 + benchmarks['^GSPC']).cumprod() * initial_capital
spx_total_return = (spx_cumulative[-1] - initial_capital) / initial_capital * 100

# Performance & Return Metrics
def calculate_metrics(strategy_returns, benchmark_returns, benchmark_name):
    # 1. Total Return
    total_return_strategy = (strategy_returns.add(1).prod() - 1) * 100
    total_return_benchmark = (benchmark_returns.add(1).prod() - 1) * 100
    
    # 2. CAGR
    num_years = len(strategy_returns) / 12
    cagr_strategy = ((1 + total_return_strategy/100) ** (1/num_years) - 1) * 100
    cagr_benchmark = ((1 + total_return_benchmark/100) ** (1/num_years) - 1) * 100
    
    # 3. Annualized Return
    annualized_strategy = strategy_returns.mean() * 12 * 100
    annualized_benchmark = benchmark_returns.mean() * 12 * 100
    
    # 4. Monthly Returns
    monthly_avg_strategy = strategy_returns.mean() * 100
    monthly_avg_benchmark = benchmark_returns.mean() * 100
    
    # 5. Cumulative Return Curve
    plt.figure(figsize=(12, 6))
    (strategy_returns.add(1).cumprod().plot(label='Strategy', color='blue'))
    (benchmark_returns.add(1).cumprod().plot(label=benchmark_name, color='green'))
    plt.title('Cumulative Return Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 6. Mean and Standard Deviation
    mean_strategy = strategy_returns.mean() * 100
    std_strategy = strategy_returns.std() * 100
    mean_benchmark = benchmark_returns.mean() * 100
    std_benchmark = benchmark_returns.std() * 100
    
    # Create summary table
    metrics = {
        'Metric': ['Total Return', 'CAGR', 'Annualized Return', 
                   'Average Monthly Return', 'Mean Monthly Return', 'Std Dev Monthly Return'],
        'Strategy': [f"{total_return_strategy:.2f}%", f"{cagr_strategy:.2f}%", 
                     f"{annualized_strategy:.2f}%", f"{monthly_avg_strategy:.2f}%",
                     f"{mean_strategy:.2f}%", f"{std_strategy:.2f}%"],
        benchmark_name: [f"{total_return_benchmark:.2f}%", f"{cagr_benchmark:.2f}%", 
                         f"{annualized_benchmark:.2f}%", f"{monthly_avg_benchmark:.2f}%",
                         f"{mean_benchmark:.2f}%", f"{std_benchmark:.2f}%"]
    }
    
    return pd.DataFrame(metrics)

# Risk & Volatility Measures
def calculate_risk_metrics(strategy_returns, benchmark_returns, benchmark_name):
    # 1. Annualized Volatility
    annual_vol_strategy = strategy_returns.std() * np.sqrt(12) * 100
    annual_vol_benchmark = benchmark_returns.std() * np.sqrt(12) * 100
    
    # 2. Skewness
    skew_strategy = skew(strategy_returns)
    skew_benchmark = skew(benchmark_returns)
    
    # 3. Kurtosis
    kurt_strategy = kurtosis(strategy_returns)
    kurt_benchmark = kurtosis(benchmark_returns)
    
    # 4. Maximum Drawdown
    def max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() * 100
    
    mdd_strategy = max_drawdown(strategy_returns)
    mdd_benchmark = max_drawdown(benchmark_returns)
    
    # 5. Average Drawdown
    def avg_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.mean() * 100
    
    avg_dd_strategy = avg_drawdown(strategy_returns)
    avg_dd_benchmark = avg_drawdown(benchmark_returns)
    
    # Create summary table
    risk_metrics = {
        'Metric': ['Annualized Volatility', 'Skewness', 'Kurtosis', 
                   'Max Drawdown', 'Avg Drawdown'],
        'Strategy': [f"{annual_vol_strategy:.2f}%", f"{skew_strategy:.2f}", 
                     f"{kurt_strategy:.2f}", f"{mdd_strategy:.2f}%", 
                     f"{avg_dd_strategy:.2f}%"],
        benchmark_name: [f"{annual_vol_benchmark:.2f}%", f"{skew_benchmark:.2f}", 
                         f"{kurt_benchmark:.2f}", f"{mdd_benchmark:.2f}%", 
                         f"{avg_dd_benchmark:.2f}%"]
    }
    
    return pd.DataFrame(risk_metrics)

# Risk-Adjusted Performance
def calculate_risk_adjusted(strategy_returns, benchmark_returns, benchmark_name, risk_free_rate=0.04):
    # Align both series
    strategy_returns, benchmark_returns = strategy_returns.align(benchmark_returns, join='inner')
    # 1. Sharpe Ratio
    excess_strategy = strategy_returns - (risk_free_rate/12)
    sharpe_strategy = (excess_strategy.mean() / strategy_returns.std()) * np.sqrt(12)
    
    excess_benchmark = benchmark_returns - (risk_free_rate/12)
    sharpe_benchmark = (excess_benchmark.mean() / benchmark_returns.std()) * np.sqrt(12)
    
    # 2. Sortino Ratio
    downside_strategy = strategy_returns[strategy_returns < 0].std()
    sortino_strategy = (excess_strategy.mean() / downside_strategy) * np.sqrt(12) if downside_strategy != 0 else np.nan
    
    downside_benchmark = benchmark_returns[benchmark_returns < 0].std()
    sortino_benchmark = (excess_benchmark.mean() / downside_benchmark) * np.sqrt(12) if downside_benchmark != 0 else np.nan
    
    # 3. Beta
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns)
    beta = covariance / variance
    
    # 4. Alpha
    alpha = (strategy_returns.mean() - (risk_free_rate/12 + beta * (benchmark_returns.mean() - risk_free_rate/12))) * 12 * 100
    
    # Create summary table
    risk_adj_metrics = {
        'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Beta', 'Alpha (annualized)'],
        'Strategy': [f"{sharpe_strategy:.2f}", f"{sortino_strategy:.2f}", 
                     f"{beta:.2f}", f"{alpha:.2f}%"],
        benchmark_name: [f"{sharpe_benchmark:.2f}", f"{sortino_benchmark:.2f}", 
                         "-", "-"]
    }
    
    return pd.DataFrame(risk_adj_metrics)

# Trade Statistics
def calculate_trade_stats(df):
    total_months = len(df)
    winning_months = len(df[df['str_return'] > 0])
    losing_months = len(df[df['str_return'] < 0])
    win_rate = winning_months / total_months * 100
    
    avg_win = df[df['str_return'] > 0]['str_return'].mean() * 100
    avg_loss = df[df['str_return'] < 0]['str_return'].mean() * 100
    
    
    trade_stats = {
        'Metric': ['Total Months', 'Winning Months', 'Losing Months', 
                   'Win Rate', 'Avg Win', 'Avg Loss'],
        'Value': [total_months, winning_months, losing_months, 
                  f"{win_rate:.2f}%", f"{avg_win:.2f}%", f"{avg_loss:.2f}%", 
               ]
    }
    
    return pd.DataFrame(trade_stats)

# Additional Metrics
def calculate_additional_metrics(strategy_returns, benchmark_returns, benchmark_name):
    # 1. VaR
    var_1_strategy = np.percentile(strategy_returns, 1) * 100
    var_5_strategy = np.percentile(strategy_returns, 5) * 100
    var_1_benchmark = np.percentile(benchmark_returns, 1) * 100
    var_5_benchmark = np.percentile(benchmark_returns, 5) * 100
    
    # 2. Worst Year
    annual_strategy = strategy_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    worst_year_strategy = annual_strategy.min() * 100
    
    annual_benchmark = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    worst_year_benchmark = annual_benchmark.min() * 100
    
    # 3. Correlation
    correlation = strategy_returns.corr(benchmark_returns)
    
    # Create summary table
    additional_metrics = {
        'Metric': ['1% VaR', '5% VaR', 'Worst Year Return', 'Correlation'],
        'Strategy': [f"{var_1_strategy:.2f}%", f"{var_5_strategy:.2f}%", 
                     f"{worst_year_strategy:.2f}%", f"{correlation:.2f}"],
        benchmark_name: [f"{var_1_benchmark:.2f}%", f"{var_5_benchmark:.2f}%", 
                         f"{worst_year_benchmark:.2f}%", f"{correlation:.2f}"]
    }
    
    return pd.DataFrame(additional_metrics)

#%% Results

nasdaq_metrics_table = pd.concat([
    calculate_metrics(df['str_return'], benchmarks['^IXIC'], 'NASDAQ'),
    calculate_risk_metrics(df['str_return'], benchmarks['^IXIC'], 'NASDAQ'),
    calculate_risk_adjusted(df['str_return'], benchmarks['^IXIC'], 'NASDAQ'),
    calculate_additional_metrics(df['str_return'], benchmarks['^IXIC'], 'NASDAQ')
], ignore_index=True)

# Calculate all metrics for S&P 500 comparison
sp500_metrics_table = pd.concat([
    calculate_metrics(df['str_return'], benchmarks['^GSPC'], 'S&P 500'),
    calculate_risk_metrics(df['str_return'], benchmarks['^GSPC'], 'S&P 500'),
    calculate_risk_adjusted(df['str_return'], benchmarks['^GSPC'], 'S&P 500'),
    calculate_additional_metrics(df['str_return'], benchmarks['^GSPC'], 'S&P 500')
], ignore_index=True)

# Show as final two tables
print("Full Metrics Table (vs NASDAQ):")
print(nasdaq_metrics_table)

print("\nFull Metrics Table (vs S&P 500):")
print(sp500_metrics_table)

# Trade Statistics
print("\nTrade Statistics")
print(calculate_trade_stats(df))

nasdaq_metrics_table.to_csv("nasdaq_metrics.csv", index=False)
sp500_metrics_table.to_csv("sp500_metrics.csv", index=False)
calculate_trade_stats(df).to_csv("trade_stats.csv", index=False)

# Plot equity curves
plt.figure(figsize=(12, 6))
(1 + df['str_return']).cumprod().plot(label='Strategy', color='blue')
(1 + benchmarks['^IXIC']).cumprod().plot(label='NASDAQ', color='green')
(1 + benchmarks['^GSPC']).cumprod().plot(label='S&P 500', color='red')
plt.title('Equity Curve Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.savefig("equity_curve.png", bbox_inches='tight')
plt.close()


