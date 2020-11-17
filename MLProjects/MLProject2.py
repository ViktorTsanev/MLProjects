# 02450: Introduction to Machine Learning & Data Mining
# Project 2 - Data: Feature extraction, and visualization
# Adrienne Cohrt s184426 & Viktor Tsanev s184453

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import OrdinalEncoder
import toolbox_02450
from toolbox_02450 import rlr_validate
import statistics as std

#################################Part A#####################################################

#Using pre-removed NA data
data = pd.read_csv('charactersNoNaN.csv')
enc = OrdinalEncoder()
# Discrete: hair color, skin color, eye color, gender, home world, species
discrCols = np.stack((data["hair_color"], data["skin_color"], data["eye_color"], data["gender"], data["homeworld"], data["species"]), axis = -1)
enc.fit(discrCols)
X_enc=enc.transform(discrCols)

# Fix Jabba the Hutt's mass (remove comma)
#data["mass"][15] = data["mass"][15].replace(',','')
#print(data["mass"][15])
# print(data)

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

y = contCols[:,1].T
print("Y")
print (y)
y = stats.zscore(y)     #feature transformation?
print("y" )
print( y)
print("mean")
print(std.mean(y))
print(standard deviation)
print (std.stdev(y))

y_label = "mass"
contCols = np.delete(contCols, 1, axis=1)
X = np.column_stack((contCols, X_enc))
# Add offset attribute
Xoff = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = Xoff.shape

# Values of lambda
lambdas = np.logspace(0, 5, 50)
K=10

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, cvf=K)

plt.title('Model error vs regularization parameter')
plt.semilogx(opt_lambda, opt_val_err, color='green', markersize=10, marker='D')
plt.loglog(lambdas, train_err_vs_lambda.T, 'b-', lambdas, test_err_vs_lambda.T, 'r-')
plt.xlabel('Regularization parameter')
plt.ylabel('Model error rate')
plt.legend(['Test minimum', 'Training error', 'Validation error'])
plt.grid()
# plt.show()

print('Regularized linear regression:')
print('- Training error: {0}'.format(train_err_vs_lambda.mean()))
print('- Test error:     {0}'.format(test_err_vs_lambda.mean()))

#################################Part B#####################################################
from sklearn import model_selection

lambdas_vect = np.empty((K,1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
# w_noreg = np.empty((M, K))

k = 0
for i_train, j_test in model_selection.KFold(K, shuffle=True, random_state = 0).split(Xoff, y):

    # extract training and test set for current CV fold
    X_train = Xoff[i_train]
    y_train = y[i_train]
    X_test = Xoff[j_test]
    y_test = y[j_test]
    internal_cross_validation = 10

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

    lambdas_vect[k] = opt_lambda

    # Display the results for the last cross-validation fold
    if k == 6:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title('Optimal lambda: 1e{0}'.format(np.round(np.log10(opt_lambda), 2)))
        plt.loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        plt.semilogx(opt_lambda, opt_val_err, color='green', markersize=10, marker='D')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error', 'Validation error'])
        plt.grid()

    k += 1

min_error = np.min(Error_test_rlr)
opt_idx = np.argmin(Error_test_rlr)
# best_weights = w_rlr[opt_idx,:]
best_lambda = lambdas_vect[opt_idx]


def best_regression_model(X_data):
    return X_data @ w_rlr[:, opt_idx]

# plt.show()
# Display results
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))


print("Training errors for regularized model are", np.round(Error_train_rlr, 2))
print("Testing errors for regularized model are", np.round(Error_test_rlr, 2))
print("Lambdas are", lambdas_vect)

print("Optimal lambda", lambdas_vect[opt_idx])