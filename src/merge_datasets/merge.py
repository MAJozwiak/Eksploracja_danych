from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.fisrt_dataset.preprocessing import preprocessing_firt_dataset
from src.second_dataset.preprocessing import preprocessing_second_dataset

df1 = preprocessing_firt_dataset()
df2 = preprocessing_second_dataset()

def merge_datasets_by_month(df1, df2):

    df1['year_month'] = df1['year'].astype(str) + '-' + df1['month'].astype(str).str.zfill(2)
    df2['year_month'] = pd.to_datetime(df2['time'], errors='coerce').dt.to_period('M').astype(str)
    df2 = df2.dropna(subset=['year_month'])
    df2_monthly = df2.groupby('year_month').mean(numeric_only=True).reset_index()
    merged_df = df1.merge(df2_monthly, on='year_month', how='left')

    #puste wartości po megrowaniu zastępowane są mediana
    for col in merged_df.columns:
        if col != 'year_month' and merged_df[col].isnull().any():
            mediana = merged_df[col].median()
            merged_df[col].fillna(mediana, inplace=True)

    #tutaj usuwane kolumna year_month zamieniana jest na string
    base_date = pd.Period('2000-01', freq='M')
    merged_df['year_month'] = merged_df['year_month'].apply(lambda x: (pd.Period(x, freq='M') - base_date).n)

    #usuwane są kolumny "year" i "month"
    if 'year' in merged_df.columns and 'month' in merged_df.columns:
        merged_df = merged_df.drop(columns=['year', 'month'])

    print("Liczba wartości null w każdej kolumnie:")
    print(merged_df.isnull().sum())
    print("\nLiczba wartości nie-null w każdej kolumnie:")
    print(merged_df.notnull().sum())

    print(merged_df)
    return merged_df

def standarization(merged_df):
    numerical_columns = ['arr_flights', 'arr_cancelled', 'arr_diverted', 'tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']
    scaler = StandardScaler()

    merged_df.loc[:, numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])
    print(merged_df.head(10))

    return merged_df


def merging():
    merged_df = merge_datasets_by_month(df1,df2)
    merged_df.describe()
    srednia_lotow = merged_df['arr_flights'].mean()
    print(f"Średnia ilość lotów na wiersz: {srednia_lotow}")
    merged_df = standarization(merged_df)

    return merged_df


