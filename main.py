#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:13:11 2024

@author: GabGren
"""
import calendar
from datetime import date
import numpy as np
import pandas as pd

import FileManagement as fm
import FilterSpace as fs
import ProcessData as prd
import TranslateAttributes as trAtt
import Transmogrifier as tm 

def daysInMonth(month):
    if not 1 <= month <= 12:
        raise ValueError("This is not a valid month number")
    # use a non-leap year so February == 28
    return calendar.monthrange(2019, month)[1]
              
# Take a loadprofile over 24 hours and distribute it to 365 days, ending up with 8760 hours              
def reshapeLoadProfile(loadProfile_df, year):
    ts_8760 = pd.read_excel('data/8760hours.xlsx')
    loadProfile_8760 = np.tile(loadProfile_df, [365,1])
    loadProfile_8760 = pd.DataFrame(data = loadProfile_8760, columns = ['hours', 'Max Power (kW)', 'Base load profile', 'Flat load profile', 'Shaved load profile'])
    loadProfile_8760 = loadProfile_8760.drop('hours', axis=1)
    loadProfile_8760 = loadProfile_8760.astype('int32')
    
    loadProfile = pd.concat([ts_8760, loadProfile_8760], axis = 1)
    loadProfile['Year'] = year
    
    return loadProfile

# Returns True for months where high load for power applies, otherwise False
def isHighLoadMonth(month):
    """
    Accepts int month (1-12) Returns True for Nov, Dec, Jan, Feb, Mar.
    """
    highLoadMonths = [11, 12, 1, 2, 3] #TODO inte säker att dessa stämmer för alla elnätsföretag, men antar detta för nu
    if month in highLoadMonths:
        return True
    else:
        return False

# Returns True for hours where high load applies, otherwise False.
# Weekends are excluded (always low load).
def isHighLoadTime(df, RE, year, month, day, hour):
    """
    df: highload dataframe (indexed by RE)
    RE: subreporting unit key
    month, day, hour: integers
    loadProfile_df: optional DataFrame for the current year (used to determine weekday)
    """
    # exclude weekends (Saturday=5, Sunday=6)
    try:
        weekday = date(int(year), int(month), int(day)).weekday()
    except Exception:
        return False
    if weekday >= 5:
        return False

    # get RE row
    REdata = df.loc[RE]

    start = REdata.get('StartHour', None)
    end = REdata.get('EndHour', None)

    if start in ('-', None, 'None', ''):
        return False

    start = int(start)
    end = int(end)

    if start == 0 and end == 0:
        return False

    if start <= hour <= end and isHighLoadMonth(month):
        return True

    return False

# 
def taxAndfixedFee_ScaledByLoad_Yearly(networkPrices_df, RE, loadProfile_df, scenario, taxesAndFixedFees_prices_df):
    taxes = networkPrices_df.loc[RE]['Myndighetsavgifter Kr, exkl. moms']
    fixedFee = networkPrices_df.loc[RE]['Fast avgift Kr, exkl. moms']

    totalKWh = loadProfile_df[scenario].sum()

    taxesAndFixedFees_prices_df[RE] = (taxes + fixedFee) /totalKWh

    return taxesAndFixedFees_prices_df

# 
def kWCharge_ScaledByLoad_Monthly(networkPrices_df, RE, month, loadProfile_df, scenario, kWCharge_prices_df):
    # Find the monthly subscribed capacity
    monthlyLoad = loadProfile_df.loc[(loadProfile_df['Month'] == month), :]
    monthlyPeaks = monthlyLoad.nlargest(3, scenario) # the three largest load hours in month (not entirely correct, also depends on the company, this is modeled after https://www.seom.se/el/elnat/effektavgiften/)
    monthlySubCap = monthlyPeaks.loc[:,scenario].mean()

    costOfMonthlySubcap = monthlySubCap * networkPrices_df.loc[RE,'Abonnerad effekt kr/kW']
    kWCharge_prices_df.loc[(loadProfile_df['Month'] == month), RE] = costOfMonthlySubcap * monthlyLoad.loc[:,scenario] / (monthlyLoad.loc[:,scenario].dot(monthlyLoad.loc[:,scenario]))

    return kWCharge_prices_df

def kWhCharge_ScaledByLoad_Hourly(networkPrices_df, highload_df, RE, month, day, hour, loadProfile_df, kWhCharge_prices_df):
    if (3 <= month <= 5) | (9 <= month <= 11):
        season = "Vår/höst"
    elif 6 <= month <= 8:
        season = "Sommar"
    elif (month == 12) | (1 <= month <= 2):
        season = "Vinter"
    else:
        raise ValueError("This is not a valid month number")
    
    # Determine high or low load time
    if isHighLoadTime(highload_df, RE, year, month, day, hour): # Höglasttid (true)
        last = 'hög'
    else: # Låglasttid (false)
        last = 'låg'
        
    kWhCharge_colName = season + ' ' + last + ' öre/kWh'
    kWhCharge = networkPrices_df.loc[RE][kWhCharge_colName] / 100 #return in kr/kWh and not öre/kWh

    hourlyLoad = loadProfile_df.loc[(loadProfile_df['Day'] == day) & (loadProfile_df['Month'] == month) & (loadProfile_df['Hour'] == hour), :]

    kWhCharge_prices_df.loc[hourlyLoad.index[0], RE] = kWhCharge
    return kWhCharge_prices_df

def calculateNetworkPrice_ScaledByLoad_RElist(networkPrices_df, highload_df, RElist, loadProfile_df, scenario, year):
    taxesAndFixedFees_prices_df = pd.DataFrame(data=loadProfile_df[['Day', 'Month', 'Year', 'Hour', 'Season']])
    kWCharge_prices_df = pd.DataFrame(data=loadProfile_df[['Day', 'Month', 'Year', 'Hour', 'Season']])
    kWhCharge_prices_df = pd.DataFrame(data=loadProfile_df[['Day', 'Month', 'Year', 'Hour', 'Season']])

    for RE in RElist:
        taxesAndFixedFees_prices_df = taxAndfixedFee_ScaledByLoad_Yearly(networkPrices_df, RE, loadProfile_df, scenario, taxesAndFixedFees_prices_df)
        for month in range(1, 13):
            kWCharge_prices_df = kWCharge_ScaledByLoad_Monthly(networkPrices_df, RE, month, loadProfile_df, scenario, kWCharge_prices_df)

            numDays = daysInMonth(month)
            for day in range(1, numDays + 1):
                for hour in range(0, 24):
                    kWhCharge_prices_df = kWhCharge_ScaledByLoad_Hourly(networkPrices_df, highload_df, RE, month, day, hour, loadProfile_df, kWhCharge_prices_df, year)

    return taxesAndFixedFees_prices_df, kWCharge_prices_df, kWhCharge_prices_df

def calculateElectricityPrice_8760(elspot_df, biddingArea, loadProfile_df, RElist):
    spot_prices_df = pd.DataFrame(data=loadProfile_df[['Day', 'Month', 'Year', 'Hour', 'Season']])
    for RE in RElist:
        spot_prices_df[RE] = elspot_df[[f'Electricity price ({biddingArea}, SEK/MWh)']]/1000
    
    return spot_prices_df

def transmogrifierInput(year, networkPrices_df, highload_df, RElist, loadProfile_df, scenario, elspot_df, biddingArea):
    taxesAndFixedFees_prices_df, kWCharge_prices_df, kWhCharge_prices_df = calculateNetworkPrice_ScaledByLoad_RElist(
        networkPrices_df, highload_df, RElist, loadProfile_df, scenario, year
    )
    spot_prices_df = calculateElectricityPrice_8760(elspot_df, biddingArea, loadProfile_df, RElist)
    return taxesAndFixedFees_prices_df, kWCharge_prices_df, kWhCharge_prices_df, spot_prices_df

#### Create scenario load profiles
def createScenarioLoadProfiles():
    loadProfile_raw_df = fm.readLoadProfile('data/EV-bus-charging-needs-Arsalan.xlsx')

    scenarioList = ['Base load profile', 'Flat load profile', 'Shaved load profile']

    ### Scenario 0: Base case - Arsalan's load profile
    loadProfile_scenarios_df = loadProfile_raw_df.copy()
    loadProfile_scenarios_df = loadProfile_scenarios_df.rename(columns={'Energy (kWh)': 'Base load profile'})

    ### Scenario 1: Flat load profile
    loadProfile_scenarios_df['Flat load profile'] = loadProfile_scenarios_df['Base load profile'].sum()/24 #~221 kWh per hour

    ### Scenario 2: Peak shaving 10%
    peak_hour = loadProfile_scenarios_df['Base load profile'].idxmax()
    peak_value = loadProfile_scenarios_df.loc[peak_hour, 'Base load profile']

    redistribute_amount = peak_value * 0.10
    loadProfile_scenarios_df['Shaved load profile'] = loadProfile_scenarios_df['Base load profile']
    loadProfile_scenarios_df.loc[peak_hour, 'Shaved load profile'] = peak_value * 0.90

    # Determine adjacent hours
    before_hour = (peak_hour - 1) % 24
    after_hour = (peak_hour + 1) % 24

    # Distribute the 10% amount to adjacent hours (split equally)
    loadProfile_scenarios_df.loc[before_hour, 'Shaved load profile'] += redistribute_amount / 2
    loadProfile_scenarios_df.loc[after_hour, 'Shaved load profile'] += redistribute_amount / 2 

    return loadProfile_scenarios_df, scenarioList

effectCustomerType = 2 # Possible 1, 2, 3
biddingArea = 'SE3'

modelingMunicipalities = ['Skövde', 'Götene', 'Skara', 'Falköping', 'Tidaholm', 'Hjo', 'Tibro', 'Töreboda', 'Mariestad']

modelingData = fm.readModelingAreas('Data')

# Create dataframes to be populated in for-loop below for visualisation
taxesAndFixedFees_prices_seasonHourly_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Hour', 'Municipality', 'Tax and Fixed Fee (SEK/kWh)'])
kWCharge_prices_seasonHourly_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Hour', 'Municipality', 'kW Fee (SEK/kWh)'])
kWhCharge_prices_seasonHourly_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Hour', 'Municipality', 'kWh Fee (SEK/kWh)'])
spot_prices_seasonHourly_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Hour', 'Municipality', 'Spot Price (SEK/kWh)'])
loadProfile_seasonHourly_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Hour', 'Load profile (kWh)'])
totalCost_allYears_df = pd.DataFrame(columns = ['Scenario', 'Year', 'Season', 'Municipality', 'Total Cost (DSO)', 'Fixed fees (DSO)', 'Power (DSO)', 'Energy (DSO)', 'Energy (Spot)'])

# Create scenario specific load profiles
loadProfile_scenarios_df, scenarioList = createScenarioLoadProfiles()

yearList = [2019, 2020, 2021, 2022, 2023]

for year in yearList:
    print(year)

    municipalityData = fs.filterMunicipalitySubset(modelingData, modelingMunicipalities, year)

    # List of REs in the modeling area
    RElist = fs.generateRElist(municipalityData, year)

    # Read in datafames with pricing data
    highload_df = fm.readHighloadtime()
    networkPrices_df = fm.readEffectCustomerPrices(effectCustomerType, year) #These are in kW
    elspot_df = fm.readElspotPrices(year, biddingArea) #SEK/kWh

    # Shape the load profile to be on 8760 hours
    loadProfile_reshape_df = reshapeLoadProfile(loadProfile_scenarios_df, year)
    for scenario in scenarioList:
        print(scenario)
        # Use scenario specific load profile
        loadProfile_df = loadProfile_reshape_df[['Day', 'Month', 'Year', 'Hour', 'Season', scenario]].copy()

        taxesAndFixedFees_prices_df, kWCharge_prices_df, kWhCharge_prices_df, spot_prices_df = transmogrifierInput(year, networkPrices_df, highload_df, RElist, loadProfile_df, scenario, elspot_df, biddingArea)

        # Clean up
        data_taxesAndFixedFees_prices_df = prd.processData(taxesAndFixedFees_prices_df)
        data_kWCharge_prices_df = prd.processData(kWCharge_prices_df)
        data_kWhCharge_prices_df = prd.processData(kWhCharge_prices_df)
        data_spot_prices_df  = prd.processData(spot_prices_df)
        data_loadProfile_df = prd.processData(loadProfile_df)

        # SEASON HOURLY (MEAN OVER ALL HOURS IN THE SEASON)
        taxesAndFixedFees_prices_seasonHourly_df = tm.seasonHourTrans(data_taxesAndFixedFees_prices_df, RElist, 'average')
        kWCharge_prices_seasonHourly_df = tm.seasonHourTrans(data_kWCharge_prices_df, RElist, 'average')
        kWhCharge_prices_seasonHourly_df = tm.seasonHourTrans(data_kWhCharge_prices_df, RElist, 'average')
        spot_prices_seasonHourly_df = tm.seasonHourTrans(data_spot_prices_df, RElist, 'average')

        #Transfer back into municipalities
        for municipality in modelingMunicipalities:
            RE = municipalityData.loc[modelingData['kommunnamn'] == municipality, f'Subredovisningsenhet ({year})'].item()

            taxesAndFixedFees_prices_seasonHourly_df[municipality] = taxesAndFixedFees_prices_seasonHourly_df[RE]
            kWCharge_prices_seasonHourly_df[municipality] = kWCharge_prices_seasonHourly_df[RE]
            kWhCharge_prices_seasonHourly_df[municipality] = kWhCharge_prices_seasonHourly_df[RE]
            spot_prices_seasonHourly_df[municipality] = spot_prices_seasonHourly_df[RE]

        def postProcessing(df, scenario, year, valueName):
            df = df.drop(RElist, axis = 1)
            df['Scenario'] = scenario
            df['Year'] = year
            df = trAtt.translateSeason(df)

            return pd.melt(df,
                           ['Scenario', 'Year', 'Season', 'Hour'],
                           modelingMunicipalities,
                           var_name = 'Municipality',
                           value_name= valueName)

        taxesAndFixedFees_prices_seasonHourly_df = postProcessing(taxesAndFixedFees_prices_seasonHourly_df, scenario, year, 'Tax and Fixed Fee (SEK/kWh)')
        kWCharge_prices_seasonHourly_df = postProcessing(kWCharge_prices_seasonHourly_df, scenario, year, 'kW Fee (SEK/kWh)')
        kWhCharge_prices_seasonHourly_df = postProcessing(kWhCharge_prices_seasonHourly_df, scenario, year, 'kWh Fee (SEK/kWh)')
        spot_prices_seasonHourly_df = postProcessing(spot_prices_seasonHourly_df, scenario, year, 'Spot Price (SEK/kWh)')
        
        # Concat the prices to one dataframe for visualization
        taxesAndFixedFees_prices_seasonHourly_allYears_df = pd.concat([taxesAndFixedFees_prices_seasonHourly_allYears_df, taxesAndFixedFees_prices_seasonHourly_df], axis = 0, ignore_index = True)
        kWCharge_prices_seasonHourly_allYears_df = pd.concat([kWCharge_prices_seasonHourly_allYears_df, kWCharge_prices_seasonHourly_df], axis = 0, ignore_index = True)
        kWhCharge_prices_seasonHourly_allYears_df = pd.concat([kWhCharge_prices_seasonHourly_allYears_df, kWhCharge_prices_seasonHourly_df], axis = 0, ignore_index = True)
        spot_prices_seasonHourly_allYears_df = pd.concat([spot_prices_seasonHourly_allYears_df, spot_prices_seasonHourly_df], axis = 0, ignore_index = True)

        # Save load profile for visualisation later
        loadProfile_vis_df = loadProfile_df.head(24).copy()
        loadProfile_vis_df['Scenario'] = scenario
        loadProfile_vis_df = loadProfile_vis_df[['Scenario', 'Year', 'Season', 'Hour', scenario]].rename(columns={scenario: 'Load profile (kWh)'})
        loadProfile_seasonHourly_allYears_df = pd.concat([loadProfile_seasonHourly_allYears_df, loadProfile_vis_df], axis = 0, ignore_index = True)

        loadProfile_864_df = np.tile(loadProfile_vis_df, [36,1])
        loadProfile_864_df = pd.DataFrame(data = loadProfile_864_df, columns = ['Scenario', 'Year', 'Season', 'Hour', 'Load profile (kWh)'])
        loadProfile_864_df['Season'] = taxesAndFixedFees_prices_seasonHourly_df['Season']
        loadProfile_864_df['Municipality'] = taxesAndFixedFees_prices_seasonHourly_df['Municipality']
     
        # Calculate and save total cost for visualisation later
        totalCost = pd.DataFrame(data=loadProfile_864_df[['Scenario', 'Municipality', 'Year', 'Season', 'Hour']])
        totalCost['Total Price (DSO)'] = taxesAndFixedFees_prices_seasonHourly_df['Tax and Fixed Fee (SEK/kWh)'] + kWCharge_prices_seasonHourly_df['kW Fee (SEK/kWh)'] + kWhCharge_prices_seasonHourly_df['kWh Fee (SEK/kWh)']
        totalCost['Total Cost (DSO)'] = totalCost['Total Price (DSO)'] * loadProfile_864_df['Load profile (kWh)']
        totalCost['Fixed fees (DSO)'] = taxesAndFixedFees_prices_seasonHourly_df['Tax and Fixed Fee (SEK/kWh)'] * loadProfile_864_df['Load profile (kWh)']
        totalCost['Power (DSO)'] = kWCharge_prices_seasonHourly_df['kW Fee (SEK/kWh)'] * loadProfile_864_df['Load profile (kWh)']
        totalCost['Energy (DSO)'] = kWhCharge_prices_seasonHourly_df['kWh Fee (SEK/kWh)'] * loadProfile_864_df['Load profile (kWh)']
        totalCost['Energy (Spot)'] = spot_prices_seasonHourly_df['Spot Price (SEK/kWh)'] * loadProfile_864_df['Load profile (kWh)']

        totalCost = totalCost.groupby(['Scenario', 'Year', 'Season', 'Municipality'])[['Total Cost (DSO)', 'Fixed fees (DSO)', 'Power (DSO)', 'Energy (DSO)', 'Energy (Spot)']].sum().reset_index()
        totalCost_allYears_df = pd.concat([totalCost_allYears_df, totalCost], axis = 0, ignore_index = True)

loadProfile_seasonHourly_allYears_df.to_csv('loadProfileAllYears.csv')
totalCost_allYears_df.to_csv('totalCostAllYears.csv')
