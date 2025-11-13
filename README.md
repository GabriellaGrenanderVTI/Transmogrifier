# Electricity Grid Price Calculator

## Overview
This project calculates electricity grid prices scaled by load profiles for different scenarios and regional entities in Sweden. It handles various components of electricity pricing including taxes, fixed fees, power charges, and energy charges.

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

## Data (inputs)
All input data files are expected in the `data/` directory. Below are the common files used by the code and the expected structure:

<!--- This works for now, but is an ugly solution. -->
- `8760hours.xlsx`
  - Purpose: timestamps for 8760 hours (non-leap year) used to expand 24‑hour profiles.
  - Expected columns (example): `Timestamp`, `Year`, `Month`, `Day`, `Hour`
  - Notes: If missing the code can generate timestamps for a given year.

<!--- This needs looking over. It is data in the project, but we should be able to take any load profile.-->
- `Load profiles` (Excel/CSV)
  - Purpose: 24‑hour example profiles to tile into 8760 (one row per hour 0–23).
  - Expected columns: `hours`, `Max Power (kW)`, `Base load profile`, `Flat load profile`, `Shaved load profile`
  - Units: profiles are in kWh per hour (or normalized values); code expects 24 rows.

<!--- Verify that we have the latest file (2025) included -->
- `Network pricing` (Excel/CSV)
  - Purpose: network (DSO) tariffs and fees per municipality / RE.
  - Required columns (examples used in code):
    - `Myndighetsavgifter Kr, exkl. moms` (authority/tax fees, SEK)
    - `Fast avgift Kr, exkl. moms` (fixed annual fee, SEK)
    - `Abonnerad effekt kr/kW` (subscribed capacity cost, SEK/kW)
    - Seasonal kWh columns like `Winter Hög öre/kWh`, `Winter Låg öre/kWh`, `Summer Hög öre/kWh`, `Shoulder Låg öre/kWh`
  - Notes: Energy rate columns are often given in öre/kWh — the code converts to SEK/kWh (divide by 100). Validate column names match exactly or adapt mapping.

<!--- This should be noted that it is incomplete atm, and will need to be looked over -->
- `High-load definitions` (Excel/CSV)
  - Purpose: define specific high-load hours per RE (overrides simple month-based rule).
  - Expected columns: index/key = RE (municipality), `StartHour`, `EndHour`
  - Behavior: If missing or values are `-`/0 the code falls back to month-based high-load months.

<!--- Add something about the vattenfall data used -->
- `Elspot / market prices` (CSV/Excel)
  - Purpose: hourly spot prices to include spot energy cost.
  - Expected columns: timestamp column and bidding-area price columns (e.g. `SE3`) or a `PriceArea` column plus `Price`.
  - Units: common formats are SEK/MWh or öre/kWh — confirm and convert to SEK/kWh before combining:
    - SEK/MWh -> divide by 1000 to get SEK/kWh.
    - öre/kWh -> divide by 100 to get SEK/kWh.

- Example file locations (repo):
  - `data/8760hours.xlsx`
  - `data/EV-bus-charging-needs-Arsalan.xlsx`
  - `data/network_prices_YYYY.xlsx`
  - `data/elspot_YYYY.csv`
  - `data/highload_definitions.xlsx`

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
- Gabriella Grenander/VTI

## Contact
[gabriella.grenander@vti.se]