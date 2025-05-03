import subprocess

def run_data_collection():
    print("Running data collection agent...")
    subprocess.run(["python", "data_collection_agent.py"])

def run_analysis_and_trading():
    print("Running analysis and trading agent...")
    subprocess.run(["python", "analysis_trading_agent.py"])

def run_backtesting():
    print("Running backtesting agent...")
    subprocess.run(["python", "backtesting_agent.py"])

def main():
    run_data_collection()
    run_analysis_and_trading()
    run_backtesting()
    print("All agents executed successfully.")

import pandas as pd
import matplotlib.pyplot as plt

def display_results():
    print("\nLoading final metrics and displaying results...")

    # Load result tables (assuming you saved them as CSVs)
    nasdaq = pd.read_csv("nasdaq_metrics.csv")
    sp500 = pd.read_csv("sp500_metrics.csv")
    trade_stats = pd.read_csv("trade_stats.csv")

    print("\n--- Full Metrics Table (vs NASDAQ) ---")
    print(nasdaq.to_string(index=False))

    print("\n--- Full Metrics Table (vs S&P 500) ---")
    print(sp500.to_string(index=False))

    print("\n--- Trade Statistics ---")
    print(trade_stats.to_string(index=False))

    # Show the equity curve image
    img_path = "equity_curve.png"
    try:
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Equity Curve Comparison")
        plt.show()
    except FileNotFoundError:
        print("\nEquity curve image not found. Please make sure backtesting_agent saves it.")

def main():
    run_data_collection()
    run_analysis_and_trading()
    run_backtesting()
    display_results()
    print("\nAll agents executed and results displayed successfully.")

if __name__ == "__main__":
    main()
