# 02450: Introduction to Machine Learning & Data Mining
# Project 2 - Data: Feature extraction, and visualization
# Adrienne Cohrt s184426 & Viktor Tsanev s184453

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import toolbox_02450
from toolbox_02450 import rlr_validate

data = pd.read_csv('charactersNoNaN.csv')
enc = OrdinalEncoder()
discrCols = np.stack((data["hair_color"], data["skin_color"], data["eye_color"], data["gender"], data["homeworld"], data["species"]), axis = -1)
enc.fit(discrCols)
# X_enc=enc.transform([['brown', 'light', 'blue', 'male', 'Naboo', 'Human' ]]).toarray()
X_enc=enc.transform(discrCols)

# Fix Jabba the Hutt's mass
# remove comma
#data["mass"][15] = data["mass"][15].replace(',','')
#print(data["mass"][15])
print(data)
mass = []
height = []
for j in range(27):
    if str(data["mass"][j]).isdigit():
        mass.append(float(data["mass"][j]))
    if str(data["height"][j]).isdigit():
        height.append(float(data["height"][j]))
# Continuous: height, mass
newMass = np.array(data["mass"])
newHeight = np.array(data["height"])
contCols = np.stack((newHeight, newMass), axis = -1)

# Discrete: name, hair color, skin color, eye color, birth year, gender, home world, species
# discrCols = np.stack((data["hair_color"], data["skin_color"], data["eye_color"], data["gender"], data["homeworld"], data["species"]), axis = -1)
# print(discrCols)
# e = OneHotEncoder()
# e.fit(discrCols)
# X_enc = e.transform(discrCols).toarray()
y = contCols[:,1].T
y = stats.zscore(y)
y_label = "mass"
# contCols = np.delete(contCols, 3, axis = 1)
X = np.column_stack((contCols, X_enc))
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = X.shape

# Values of lambda
lambdas = np.logspace(0, 5, 50)

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, cvf=10)

# plt.figure(figsize=(8, 8))

plt.title('Model error vs regularization parameter')
plt.semilogx(opt_lambda, opt_val_err, color='green', markersize=10, marker='D')
plt.loglog(lambdas, train_err_vs_lambda.T, 'b-', lambdas, test_err_vs_lambda.T, 'r-')
plt.xlabel('Regularization parameter')
plt.ylabel('Model error rate')
plt.legend(['Test minimum', 'Training error', 'Validation error'])
plt.grid()
plt.show()

print('Regularized linear regression:')
print('- Training error: {0}'.format(train_err_vs_lambda.mean()))
print('- Test error:     {0}'.format(test_err_vs_lambda.mean()))
