# Cup and Handle Pattern Detection System

Automated detection and visualization of Cup and Handle trading patterns in cryptocurrency data using Python.

## Quick Setup

### 1. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Install TA-Lib
**Windows:** Download `.whl` from [TA-Lib releases](https://github.com/TA-Lib/ta-lib-python/releases)
```bash
pip install TA_Lib-0.6.5-cp3x-cp3x-win_amd64.whl
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```bash
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib && ./configure --prefix=/usr && make && sudo make install
pip install TA-Lib
```

## Usage

1. **Add CSV files** to `data/` directory with columns: `open_time,open,high,low,close,volume`
2. **Run detection:**
```bash
python main.py
```

## Output
- **Charts:** `patterns/cup_handle_*.png`
- **Reports:** `reports/pattern_summary_report.csv`

## Data Format
```csv
open_time,open,high,low,close,volume
1640995200000,47500.00,47650.00,47450.00,47600.00,125.45
```

## Troubleshooting
- **TA-Lib errors:** Follow platform-specific installation above
- **No patterns found:** Ensure data has 1000+ rows and realistic price movements
- **Import errors:** Check virtual environment activation and dependencies