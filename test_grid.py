import pandas as pd
import numpy.random as rand 
from rbf import *
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.cross_validation import KFold

data = pd.read_csv("sin.csv",";") 
X = data[data.columns[:-1]] 
y = data[data.columns[-1]]

X = X.as_matrix()
y = y.as_matrix()

score = metrics.make_scorer(mean_squared_error) 
tuned_parameters = [ {'k' : [10], 
                      'compute_widths' : [False], 
                      'p' : np.logspace(-6,3,10)}]

gridsearch = GridSearchCV(RBFNet(),
                          tuned_parameters,
                          n_jobs = 10, pre_dispatch = "n_jobs",
                          cv = KFold(len(X),n_folds=5)) 
gridsearch.fit(X,y) 
print(gridsearch.best_params_)

model = RBFNet()
model.set_params(**gridsearch.best_params_) 
model.fit(X,y) 
yy = model.predict(X)

err = 100*mean_squared_error(y,yy)
print("Err:", err)
