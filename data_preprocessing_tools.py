import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Data.csv")
# Features
x = dataset.iloc[:, :-1].values
# Dependent variable vector
y = dataset.iloc[:, -1:].values

# Handling the missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# Encoding the Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
