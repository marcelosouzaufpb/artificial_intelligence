# %pylab inline
from io import StringIO
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


# Lidando com dados ausentes - Removendo exemplos ou fetures
# simulando como se estivessemos abrindo um arquivo csv

csv_data = '''Atrib1, Atrib2, Atrib3, Atrib4
            11.1, 21.2, 33.3, 40.4
            15.5, 26.0,, 48.0
            16.0, 21.0, 32.2'''

# df = pd.read_csv(StringIO(csv_data))
# df.info()

# Procurando dados daltantes com isna() ou isnull()
# df.isna()
# df.isnull()
# df.isnull().sum()

# Removendo os exemplos(linhas) contendo algum dado faltante
# df.dropna()
# # df.dropna(axis=0)
# df.dropna(axis=1)


csv_data = '''Atrib1,Atrib2,Atrib3,Atrib4,Atrib5
11.1,21.2,33.3,40.4
15.5,26.0,,48.0
16.0,21.0,32.2
,,,,
'''

df = pd.read_csv(StringIO(csv_data))
# # df.dropna(how='all')
# # df.dropna(how='all', axis=1)
# df.dropna(thresh=df.columns.size-1)
# df.info()

imputer = SimpleImputer(strategy='maen')
imputer = imputer.fit(df)
imputer_data = imputer.fit(df)
imputer_data = imputer.transform(df.values)
print(imputer_data)


