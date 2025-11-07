# Electricity Grid Price Calculator

## Overview
This project calculates electricity grid prices scaled by load profiles for different scenarios
and regional entities in Sweden. It handles various components of electricity pricing including
taxes, fixed fees, power charges, and energy charges.

## Features
- Calculate grid prices for multiple municipalities
- Support for different load profile scenarios:
  - Base load profile
  - Flat load profile
  - Peak-shaved load profile
- Price components:
  - Taxes and fixed fees (yearly basis)
  - Power charges (monthly basis)
  - Energy charges (hourly basis)
  - Spot prices
- Seasonal and time-of-use pricing
- High load period handling
- Data visualization capabilities

## Prerequisites
- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - openpyxl (for Excel file handling)

## Project Structure
```
Transmogrifier/
│
├── GridPriceScaledByLoad.py    # Main calculation module
├── FileManagement.py           # File I/O operations
├── FilterSpace.py              # Data filtering functions
├── ProcessData.py             # Data processing utilities
├── TranslateAttributes.py     # Data translation helpers
├── data/                      # Input data directory
│   ├── 8760hours.xlsx
│   ├── EV-bus-charging-needs-Arsalan.xlsx
│   └── ...
└── README.md
```

## Usage
1. Ensure all input data files are present in the `data/` directory
2. Configure parameters in GridPriceScaledByLoad.py:
   - EFFECT_CUSTOMER_TYPE
   - BIDDING_AREA
   - MODELING_MUNICIPALITIES
   - YEAR_LIST

3. Run the main script:
```bash
python GridPriceScaledByLoad.py
```

## Output
The script generates two CSV files:
- loadProfileAllYears.csv: Contains load profiles for all scenarios
- totalCostAllYears.csv: Contains cost breakdowns for all price components

## Configuration
### Customer Types
- EFFECT_CUSTOMER_TYPE options:
  - 1: Small customers
  - 2: Medium customers
  - 3: Large customers

### Bidding Areas
- Available areas: SE1, SE2, SE3, SE4
- Default: SE3

## Data Requirements
- Network pricing data (yearly)
- High load time definitions
- Load profiles
- Spot prices
- Municipality data

## License
[Your license information here]

## Contributors
- [Gabriella Grenander/VTI]

## Contact
[gabriella.grenander@vti.se]