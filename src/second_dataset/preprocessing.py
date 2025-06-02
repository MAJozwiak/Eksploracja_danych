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

    #fill NaN with mean
    df.isnull().sum()
    df['tavg'] = df['tavg'].fillna(df['tavg'].mean())
    df['wdir'] = df['tavg'].fillna(df['tavg'].mean())
    df['pres'] = df['pres'].fillna(df['pres'].mean())
    df.isnull().sum()
    return df


# def standarization(df):
#     numerical_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']
#     scaler = StandardScaler()
#
#     df.loc[:, numerical_columns] = scaler.fit_transform(df[numerical_columns])
#
#     print(df.describe())
#     return df

def preprocessing_second_dataset():
    df = read_file()
    df = data_cleaning(df)
    #standarization(df)
    return df