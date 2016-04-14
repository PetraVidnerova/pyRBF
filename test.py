import pandas as pd 
import numpy.random as rand 
from rbf import *
from sklearn.metrics import mean_squared_error


data = pd.read_csv("sin.csv",";") 
X = data[data.columns[:-1]] 
y = data[data.columns[-1]]

X = X.as_matrix()
y = y.as_matrix()

model = RBFNet(10,p=0.1,compute_widths='none')
model.fit(X,y) 
yy = model.predict(X)

err = 100*mean_squared_error(y, yy)
print("Err:", err)

from sklearn.externals import joblib 

joblib.dump(model, "rbf.pkl")
model2 = joblib.load("rbf.pkl") 

yy = model2.predict(X)
err = 100*mean_squared_error(y, yy)
print("Err after load:", err) 
