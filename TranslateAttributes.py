##################### TRANSLATE #####################

def translateSeason(df):
    """
    Translate the 'Season' column from numeric to string representation.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Season' column
        
    Returns:
        pd.DataFrame: DataFrame with translated 'Season' column
    """
    season_map = {
        1: 'Winter',
        2: 'Spring',
        3: 'Summer',
        4: 'Autumn'
    }
    df['Season'] = df['Season'].map(season_map)
    
    return df

def translateHour(df):
    """
    Translate the 'Hour' column from numeric to 'Hxx' string representation.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Hour' column
        
    Returns:
        pd.DataFrame: DataFrame with translated 'Hour' column
    """
    for i in range(24):
        df.loc[df['Hour'] == i, 'Hour'] = 'H' + str(i).zfill(2)
    
    return df