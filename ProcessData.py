import pandas as pd

def createDatetime(df):
    """
    Create a datetime column from Year, Month, Day, and Hour columns.

    Args:
        df (pd.DataFrame): DataFrame containing 'Year', 'Month', 'Day', 'Hour' columns
        
    Returns:
        pd.DataFrame: DataFrame with added 'Date' column
    """
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    return df

def createSeasonColumn(df):
    """
    Create a 'Season' column based on the month in the 'Date' column.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date' column
    
    Returns:
        pd.DataFrame: DataFrame with added 'Season' column
    """
    # 1: Winter
    # 2: Spring
    # 3: Summer
    # 4: Autumn
    df['Season'] = df['Date'].dt.month%12 // 3 + 1
    
    return df

def createWeekdayColumn(df):
    """
    Create a 'Day of week' column based on the 'Date' column.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date' column

    Returns:
        pd.DataFrame: DataFrame with added 'Day of week' column
    """
    df['Day of week'] = df['Date'].dt.weekday
    
    return df

def processData(df):
    """
    Process the DataFrame by creating 'Date', 'Season', and 'Day of week' columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Year', 'Month', 'Day', 'Hour' columns
    
    Returns:
        pd.DataFrame: Processed DataFrame with added columns
    """
    df = createDatetime(df)
    df = createSeasonColumn(df)
    df = createWeekdayColumn(df)

    df.insert(0, 'Day of week', df.pop('Day of week'))
    df.insert(0, 'Season', df.pop('Season'))
    df.insert(0, 'Date', df.pop('Date'))

    df = df.drop(['Day', 'Month', 'Hour'], axis = 1) #Leave in year for visualisation

    return df