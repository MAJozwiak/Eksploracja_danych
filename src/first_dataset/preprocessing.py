import pandas as pd

def read_file():
    path = r"C:\Users\marta\PycharmProjects\Eksploracja_danych\data\Airline_Delay_Cause (1).csv"
    df = pd.read_csv(path)
    df = pd.DataFrame(df)
    print(df.head())

    return df

def filter_Boston_airports(df):
    print(df[df['airport_name'].str.contains('Boston')].describe())

    mask = df['airport_name'].str.contains("Boston", case=False, na=False)
    unikalne = df.loc[mask, 'airport_name'].nunique()
    print(f"Unique value which contains 'Boston': {unikalne}")
    print(f"Airport name with 'Boston':{df.loc[mask, 'airport_name'].unique()}")

    df = df.drop(columns=['airport'])
    df = df[df['airport_name'] == 'Boston, MA: Logan International']
    df = df.drop(columns=['airport_name'])

    return df

def data_cleaning(df):
    print(df.dtypes)
    df.isnull().sum()
    df = df.dropna(subset=['arr_del15'])

    df = df.drop(columns=['arr_delay'])
    df = df.drop(columns=['carrier_delay'])
    df = df.drop(columns=['late_aircraft_delay'])
    df = df.drop(columns=['weather_delay'])
    df = df.drop(columns=['nas_delay'])
    df = df.drop(columns=['security_ct'])
    df = df.drop(columns=['late_aircraft_ct'])
    df = df.drop(columns=['weather_ct'])
    df = df.drop(columns=['carrier_ct'])
    df = df.drop(columns=['nas_ct'])


    print(df.isnull().sum())
    return df

def vectorization(df):
    print(df['carrier'].nunique())
    print(df['carrier_name'].nunique())
    print(df[['carrier', 'carrier_name']].isnull().sum())
    df = df.drop(columns=['carrier_name'])
    categorical_columns = ['carrier']
    df = pd.get_dummies(df, columns=categorical_columns, dtype=int, drop_first=True)

    return df

def preprocessing_firt_dataset():
    df = read_file()
    df = filter_Boston_airports(df)
    df = data_cleaning(df)
    df = vectorization(df)
    print(df.head)

    return df


