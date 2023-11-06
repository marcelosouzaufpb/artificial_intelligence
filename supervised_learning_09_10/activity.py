import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder


def load_data():
    data = '''imposto,renda_media,estradas_pav,populacao_cnb,consumo
extr-high,3571.0,1976.0,0.525,541.0
extr-high,40925.0,1250.0,0.572,524.0
extr-high,3865.0,1586.0,0.580,561.0
med,4870.0,2351.0,0.529,414.0
high,4399.0,431.0,0.544,410.0
max,5342.0,1333.0,NaN,457.0
high,5319.0,11868.0,0.451,344.0
high,5126.0,2138.0,0.553,467.0
high,4447.0,8577.0,0.529,464.0
low,4512.0,8507.0,0.552,498.0
high,4391.0,5939.0,0.530,580.0
low,3560.0,1975.0,0.533,NaN
med,5126.0,14186.0,0.525,471.0
low,4817.0,6930.0,0.574,525.0
low,4207.0,5580.0,0.545,508.0
low,4332.0,8159.0,0.608,566.0
low,4318.0,10340.0,0.586,635.0
low,4206.0,8508.0,0.572,603.0
low,3718.0,4725.0,0.540,714.0
low,4716.0,5915.0,0.724,865.0
very-high,4341.0,6010.0,0.677,640.0
low,4593.0,7834.0,0.663,649.0
high,4983.0,602.0,0.602,540.0
extr-high,4897.0,2449.0,0.511,464.0
extr-low,NaN,NaN,NaN,581.0
extr-high,4258.0,4686.0,0.517,547.0
very-high,4574.0,2619.0,0.551,460.0
extr-high,3721.0,4746.0,0.544,466.0
high,3446.0,5399.0,0.548,577.0
med,3846.0,9061.0,0.579,631.0
high,4188.0,5975.0,0.563,574.0
extr-high,3601.0,4650.0,0.493,534.0
low,3640.0,6905.0,0.518,571.0
low,3333.0,6594.0,0.513,554.0
high,3063.0,5524.0,0.578,577.0
med,3357.0,4121.0,0.547,528.0
high,3528.0,3495.0,0.487,487.0
very-low,3802.0,7834.0,0.566,640.0
min,4045.0,17782.0,0.566,640.0
low,3897.0,6385.0,0.586,704.0
very-high,3535.0,3274.0,0.663,648.0
low,4345.0,3905.0,0.672,968.0
low,4449.0,4639.0,0.626,587.0
low,3856.0,3985.0,0.563,699.0
low,4300.0,3635.0,0.603,632.0
low,3745.0,2611.0,0.508,591.0
extr-low,5215.0,302.0,0.672,782.0
extr-high,4476.0,3942.0,0.571,510.0
low,4296.0,4083.0,0.623,610.0
low,5002.0,9794.0,0.583,524.0 '''
    csv_raw = StringIO(data)
    return pd.read_csv(csv_raw)


def print_basic_statistics_info(data):
    print(data.isnull().sum(), '\n')


def remove_problematic_rows(data):
    return data.dropna(thresh=3)


def remove_rows_with_invalid_label(data, column_name):
    return data.dropna(subset=[column_name])


def fill_missing_data_with_mean(data, column_name):
    imputer = SimpleImputer(strategy='mean')
    imputer = imputer.fit(data[[column_name]])
    data[column_name] = imputer.transform(data[[column_name]])
    return data


def encode_columns_with_ordinal_values(data, column_name, categories=[[]]):
    encoder = OrdinalEncoder(categories=categories)
    data[column_name] = encoder.fit_transform(data[[column_name]])
    return data


def train_decision_tree(data, label_column):
    x = data.drop(label_column, axis=1)
    y = data[label_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=4)

    estimator = DecisionTreeRegressor(max_depth=3, min_samples_split=3)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)

    df_comparison = pd.DataFrame({'Predicted': y_pred, 'Expected': y_test})
    print(df_comparison)

    print('\nMean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
    print('\nMean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    data = load_data()
    print_basic_statistics_info(data)

    data = remove_problematic_rows(data)
    print_basic_statistics_info(data)

    data = remove_rows_with_invalid_label(data, 'consumo')
    print_basic_statistics_info(data)

    data = fill_missing_data_with_mean(data, 'populacao_cnb')
    print_basic_statistics_info(data)

    data = encode_columns_with_ordinal_values(data, 'imposto', [['min', 'extr-low', 'very-low', 'low', 'med', 'high', 'very-high', 'extr-high', 'max']])

    train_decision_tree(data, 'consumo')
