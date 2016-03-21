import pandas as pd 
import numpy.random as rand 
from rbf import *


data = pd.read_csv("sin.csv",";") 
X = data[data.columns[:-1]] 
y = data[data.columns[-1]]


model = RBFNet(10,0.1)
model.fit(X,y) 
yy = model.predict()
