import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
# Features
x = dataset.iloc[:, :-1].values
# Dependent variable vector
y = dataset.iloc[:, -1:].values

print(dataset)
print(x)
print(y)
