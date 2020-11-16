# 02450: Introduction to Machine Learning & Data Mining
# Project 1 - Data: Feature extraction, and visualization
# Adrienne Cohrt s184426 & Viktor Tsanev s184453

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import toolbox_02450
from toolbox_02450 import rlr_validate

data = pd.read_csv('characters.csv')
# Fix Jabba the Hutt's mass
# remove comma
data["mass"][15] = data["mass"][15].replace(',','')
print(data["mass"][15])

# Continuous: height, mass
contCols = np.stack((data["height"], data["mass"]), axis = -1)

# Discrete: name, hair color, skin color, eye color, birth year, gender, home world, species
discrCols = np.stack((data["name"], data["hair_color"], data["skin_color"], data["eye_color"], data["birth_year"], data["gender"], data["homeworld"], data["species"]), axis = -1)
print(discrCols)
e = OneHotEncoder()
e.fit(discrCols)
X_enc = e.transform(discrCols).toarray()
y = discrCols[:,1].T
y = stats.zscore(y)
y_label = "mass"
contCols = np.delete(contCols, 3, axis = 1)
X = np.column_stack((contCols, X_enc))
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = X.shape
"""attributeNames = np.array(['Offset', 'age', 'trestbps', 'chol', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')"""

K = 10

# Values of lambda
lambdas = np.logspace(0, 5, 50)

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, cvf=K)

plt.figure(figsize=(8, 8))

plt.title('Model error vs regularization factor (lambda)')
plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
plt.loglog(lambdas, train_err_vs_lambda.T, 'b-', lambdas, test_err_vs_lambda.T, 'r-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate')
plt.legend(['Test minimum', 'Training error', 'Validation error'])
plt.grid()
plt.show()

print('Regularized linear regression:')
print('- Training error: {0}'.format(train_err_vs_lambda.mean()))
print('- Test error:     {0}'.format(test_err_vs_lambda.mean()))
