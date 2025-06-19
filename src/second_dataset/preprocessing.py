import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_file():
    path = r"C:\Users\marta\PycharmProjects\Eksploracja_danych\data\boston_weather_data.csv"
    df = pd.read_csv(path)
    df = pd.DataFrame(df)
    print(df.head())
    df.describe()
    return df

def data_cleaning(df):

    df['time'].value_counts().sum()
    print(df.dtypes)
    print(df.shape)

    df.isnull().sum()
    df['tavg'] = df['tavg'].fillna(df['tavg'].mean())
    df['wdir'] = df['tavg'].fillna(df['tavg'].mean())
    df['pres'] = df['pres'].fillna(df['pres'].mean())
    df.isnull().sum()
    return df

def preprocessing_second_dataset():
    df = read_file()
    df = data_cleaning(df)
    return df