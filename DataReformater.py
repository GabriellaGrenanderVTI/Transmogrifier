#%% 

import pandas as pd
import FileManagement as fm

years = [2019]#, 2020, 2021, 2022, 2023, 2024]#
biddingAreas = ['SE1', 'SE2', 'SE3', 'SE4']

for year in years:
    print(year)
    df = pd.DataFrame()
    for biddingArea in biddingAreas:
        df = pd.concat([df,fm.readElspotPrices_Vattenfall_test(year, biddingArea)], axis=1)

    print(df.index)
    
    # save to CSV
    print("csv:", df.index.min(), "->", df.index.max())
    df.to_csv(f"elspot_{year}.csv")
