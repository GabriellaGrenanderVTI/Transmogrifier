from datetime import datetime
from pathlib import Path
import pandas as pd

def writeDfToCsv(df, fileName):
    output_folder = Path('output')

    df.to_csv(output_folder / fileName)

def readDfFromCsv(filePath):
    return pd.read_csv(filePath)

def writeLoadAndCostToExcel(loadProfile, gridCost, fileName):
    output_folder = Path('output')

    file_name = 'export_loadAndCost_' + fileName + '_' + datetime.today().strftime('%Y%m%d') + '.xlsx'
    
    with pd.ExcelWriter(output_folder / file_name) as writer:  
        loadProfile.to_excel(writer, sheet_name='gridCost')
        gridCost.to_excel(writer, sheet_name='loadProfile')

def readEffectCustomerPrices(effectCustomerType, year):
    df = pd.read_excel('data/effektkunder_2011-2023.xlsx', header = [0,1,2])
    
    # Keep the columns to combine with result later on
    df2 = df[['REnummer', 'Län', 'Org.nr', 'Nätföretag', 'REnamn']]
    df2 = df2.droplevel([1,2], axis=1)
    
    # Find the columns that contain information about desired year
    df = df.iloc[:, df.columns.get_level_values(2) == year]
    df = df.droplevel([2], axis=1)
    
    # Find the columnd that contain information about desired customer type (3 kinds)
    df = df.iloc[:, df.columns.get_level_values(0).str.contains('NT9' + str(effectCustomerType))]
    df = df.droplevel([0], axis=1)
    colNames = list(df.columns.values)
    
    # Join the "index columns" and the desired data columns
    df = pd.concat([df2, df], axis = 1, join = 'inner')

    # Remove rows that do not contain any data and reindex
    df.dropna(axis = 0, subset = colNames , how = 'all', inplace = True)
    df = df.reset_index()
    
    # Create unique identifier for the networkconcession
    col = df['REnummer'].fillna('') + df['REnamn'].fillna('')
    df.insert(5, 'RE', col)
    df = df.set_index('RE')    
    df = df.drop(columns = ['Län', 'index'])
    return df

def readElspotPrices(year, biddingArea):
    if year in [2019, 2023, 2024]:
        return readElspotPrices_Vattenfall(year, biddingArea)
    elif year in [2020, 2021, 2022]:
        return readElspotPrices_ElspotNu(year, biddingArea)
    else:
        raise ValueError("This is not a valid year for electricity data")

def readElspotPrices_ElspotNu(year, biddingArea): #2020, 2021, 2022
    df = pd.read_excel(f'data/elspot_prices/elspot-prices_{year}_hourly_sek_8760.xlsx')

    print(df.columns)

    df.rename(columns={biddingArea: f'Electricity price ({biddingArea}, SEK/MWh)'}, inplace=True)
    df = df[[f'Electricity price ({biddingArea}, SEK/MWh)']]

    return df

def readElspotPrices_Vattenfall(year, biddingArea): # 2019, 2023
    columnNames = ['Tidsperiod', 'Pris (öre/kWh)']
    dataFolder = Path(f'data/elspot_prices/Vattenfall-data/{year}/{biddingArea}')

    parts = []

    # Read the base file
    base = pd.read_excel(dataFolder / 'data.xlsx', header=0, usecols=[0,1])
    base.columns = columnNames
    base['Tidsperiod'] = pd.to_datetime(base['Tidsperiod'], format='%Y-%m-%d %H:%M')
    parts.append(base)

    # Read weekly files
    for i in range(1, 53):
        df_i = pd.read_excel(dataFolder / f'data ({i}).xlsx', header=0, usecols=[0,1])
        df_i.columns = columnNames
        df_i['Tidsperiod'] = pd.to_datetime(df_i['Tidsperiod'], format='%Y-%m-%d %H:%M')

        # Keep only the current year
        df_i = df_i[df_i['Tidsperiod'].dt.year == year]

        parts.append(df_i)

    # Concatenate all parts
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=['Tidsperiod'])
    df['Tidsperiod'] = df['Tidsperiod'].dt.strftime('%Y-%m-%d %H:%M')

    # Adjust for lost hour due to summertime (wintertime hour is already included in data)
    if year == 2019:
        val = df.loc[df['Tidsperiod'] == '2019-03-31 01:00', 'Pris (öre/kWh)'].iloc[0]
        summertimeRow = pd.DataFrame(columns = df.columns, data = [['2019-03-31 02:00', val]]) #take same price as 01:00
    elif year == 2023:
        val = df.loc[df['Tidsperiod'] == '2023-03-26 01:00', 'Pris (öre/kWh)'].iloc[0]
        summertimeRow = pd.DataFrame(columns = df.columns, data = [['2023-03-26 02:00', val]]) #take same price as 01:00
    elif year == 2024:
        val = df.loc[df['Tidsperiod'] == '2024-03-31 01:00', 'Pris (öre/kWh)'].iloc[0]
        summertimeRow = pd.DataFrame(columns = df.columns, data = [['2024-03-31 02:00', val]]) #take same price as 01:00
    df = pd.concat([df, summertimeRow], axis=0)
    df = df.sort_values(by='Tidsperiod').reset_index(drop = True)

    df[f'Electricity price ({biddingArea}, SEK/MWh)'] = df['Pris (öre/kWh)']*10 # multiply by 10 to get SEK/MWh
    df = df.drop(columns=['Pris (öre/kWh)'])
    return df

def readElspotPrices_Vattenfall_test(year, biddingArea): # 2019, 2023
    columnNames = ['Tidsperiod', 'Pris (öre/kWh)']
    dataFolder = Path(f'data/elspot_prices/Vattenfall-data/{year}/{biddingArea}')
    
    parts = []

    # Read the base file
    base = pd.read_excel(dataFolder / 'data.xlsx', header=0, usecols=[0,1])
    base.columns = columnNames
    base['Tidsperiod'] = pd.to_datetime(base['Tidsperiod'], format='%Y-%m-%d %H:%M')
    parts.append(base)

    # Read weekly files
    for i in range(1, 53):
        df_i = pd.read_excel(dataFolder / f'data ({i}).xlsx', header=0, usecols=[0,1])
        df_i.columns = columnNames
        df_i['Tidsperiod'] = pd.to_datetime(df_i['Tidsperiod'], format='%Y-%m-%d %H:%M')

        # Keep only the current year
        df_i = df_i[df_i['Tidsperiod'].dt.year == year]

        parts.append(df_i)

    # Concatenate all parts
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=['Tidsperiod'])
    df = df.sort_values('Tidsperiod')

    # Use datetime index
    df = df.set_index('Tidsperiod')

    # Localize to Oslo to resolve DST issues
    df = df.tz_localize("Europe/Oslo",
                        nonexistent="shift_forward",    # handle spring forward
                        ambiguous="NaT")                # handle autumn back

    # Convert to UTC (no DST)
    df = df.tz_convert("UTC")

    # --- Force exactly 8760 hours ---
    full_index = pd.date_range(f"{year}-01-01 00:00",
                               f"{year}-12-31 23:00",
                               freq="H",
                               tz="UTC")
    df = df.reindex(full_index)

    # For the spring DST hour, fill with the values from next hour
    df = df.fillna(method="bfill")

    # Convert prices to SEK/MWh
    df[f"Electricity price ({biddingArea}, SEK/MWh)"] = df["Pris (öre/kWh)"] * 10
    df = df.drop(columns=["Pris (öre/kWh)"])

    # Convert back to local time for Excel (timezone-unaware)
    df = df.tz_localize(None) #.tz_convert("Europe/Oslo")

    # # Drop leap day rows
    # df = df[~((df.index.month == 2) & (df.index.day == 29))]

    return df

def readHighloadtime():
    df = pd.read_excel('data/Highloadtime.xlsx', sheet_name = 'filteredHighloadTime')
    df = df.set_index('RE')    
    return df

def readLoadProfile(path):
    return pd.read_excel(path)

def readModelingAreas(sheet):
    return pd.read_excel('data/ModelingAreas.xlsx', sheet_name=sheet)
    
def readNetworkConcessionData(year):
    if year == 2023:
        df = pd.read_excel("data/Koncessioner/Natkoncessioner_per_kommun_med_kontaktuppgifter_fixad_2023.xlsx")
        return df[['Kommunkod', 'Kommunnamn', 'Koncession', 'Spänning', 'Red.enhet', 'Företagsn']]
    elif year == 2024:
        df = pd.read_csv("data/Koncessioner/KommunKoncession_med_kontaktuppgifter_2024.csv")
        return df[['kommunkod', 'kommunnamn', 'KONCESSION', 'Spanning', 'Enhet', 'Företagsnamn']]
    else:
        raise ValueError("This is not a valid year for network concession data")