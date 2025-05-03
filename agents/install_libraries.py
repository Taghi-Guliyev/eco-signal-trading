import subprocess
import sys

required_packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "xgboost",
    "scikit-learn",
    "ta",
    "yfinance",
    "meteostat",
    "pandas_datareader",
    "scipy"
]

def install_packages():
    for package in required_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages()

