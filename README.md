#Climate-Aware Multi-Agent Quant Trading System

This repository presents a multi-agent quantitative trading system that integrates alternative climate data, macroeconomic indicators, and technical signals to generate alpha on the equity of Archer Daniels Midland (ADM). The strategy is based on machine learning (XGBoost classifier), structured into modular agents for data collection, signal generation, and backtesting.

<pre> ```text ┌────────────────────────┐ │ Data Collection Agent│ └────────────┬───────────┘ ↓ ┌───────────────────────────┐ │ Analysis & Trading Agent │ ← ML Signal via XGBoost └────────────┬──────────────┘ ↓ ┌────────────────────────────┐ │ Backtesting Agent │ ← Evaluation vs NASDAQ & S&P 500 └────────────────────────────┘ ``` ```text climate-trading-system/ │ ├── agent/ │ ├── main.py │ ├── data_collection_agent.py │ ├── analysis_trading_agent.py │ ├── backtesting_agent.py │ └── install_libraries.py │ ├── data/ │ ├── fert.xlsx │ └── drought.csv │ ├── README.md └── .gitignore ``` </pre>

🌍 Project Motivation

The use of climate-sensitive alternative data is increasingly relevant for companies like ADM, which are highly exposed to agriculture and weather volatility. This system tests whether integrating variables like drought severity, climate stress, and fertilizer prices into financial signals can enhance return and reduce risk.

🧾 Dataset Overview

The model uses the following variables:

Technical Indicator: RSI (Relative Strength Index)

Macroeconomic Indicators:

    CPI (Food inflation) 

    USD Index (DXY)

    USD Index Returns

Alternative Data:

    Fertilizer Index

    Drought Severity Index

    Climate Stress Index (Temp - Precip anomaly)

Engineered Variables:

    Climate Signal (1 if CSI > 22, -1 if < -22, else 0)

    Interaction Term = Climate Stress / Drought Severity

⚙️ How It Works
1. Data Collection Agent (data_collection_agent.py)

    Pulls ADM stock prices from Yahoo Finance.

    Fetches CPI data from FRED and USD index from Yahoo.

    Aggregates and engineers climate metrics using Meteostat and USDA datasets:

        Monthly precipitation and temperature from Illinois, Iowa, and Nebraska

        Drought index from USDA (D1–D4 severity scale)

        Fertilizer data from FAO Excel file

All features are merged into a single CSV (final_merged_features.csv).

2. Analysis & Trading Agent (analysis_trading_agent.py)

    Loads the feature file.

    Performs:

        Target engineering: Direction of ADM monthly return

        Standard scaling

        Training an XGBoost Classifier using pre-2020 data

        Prediction on post-2020 test data

    Outputs:

        Prediction accuracy and confusion matrix

        Strategy returns vs buy-and-hold

3. Backtesting Agent (backtesting_agent.py)

    Calculates:

        Performance metrics (CAGR, Sharpe, etc.)

        Risk metrics (Drawdown, VaR)

        Risk-adjusted alpha vs NASDAQ and S&P 500

    Plots:

        Equity curve comparisons

        Outputs key metric tables and trade statistics

🧰 Setup Instructions
1. Clone the Repo

   git clone https://github.com/Taghi-Guliyev/eco-signal-trading.git

   cd eco-signal-trading

3. Install Required Packages

Run the helper script to install necessary libraries:
    
    python install_libraries.py

3. Prepare Required Files

Ensure the following files are manually downloaded and placed in the same directory as the agents:

- fert.xlsx	    (Fertilizer index data)

- drought.csv  	(Drought index from USDA)

- All .py files	(Agent scripts)

▶️ Running the System

You can run the entire pipeline with:

python main_agent.py

This will sequentially execute:

    Data collection

    XGBoost prediction

    Backtesting and metrics display


📊 Backtest Summary on test data

    Accuracy: 74.60% (Win Rate)

    Total Return: +593.63% (between 01/2020 - 03/2025)

    Sharpe Ratio: 1.37

    Alpha vs NASDAQ: +30.92% (annualized)

    Alpha vs S&P 500: +30% (annualized)


📁 File Structure

<pre> ```text climate-trading-system/ │ ├── agent/ │ ├── main.py │ ├── data_collection_agent.py │ ├── analysis_trading_agent.py │ ├── backtesting_agent.py │ └── install_libraries.py │ ├── data/ │ ├── fert.xlsx │ └── drought.csv │ ├── README.md └── .gitignore ``` </pre>


📄 License

This project is for educational and research purposes.





