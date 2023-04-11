# Packages
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training and Testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
