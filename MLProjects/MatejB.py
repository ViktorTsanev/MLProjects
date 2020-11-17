import numpy as np
from scipy.stats import stats
from sklearn.preprocessing import OneHotEncoder
import toolbox_02450
from toolbox_02450 import rlr_validate
import matplotlib.pyplot as plt
from sklearn import model_selection

  
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

X_wo = df[:-1].to_numpy()

y_attr = 'target'
cont_attr = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sel_attr = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'thal']

X_sel_wo = df[sel_attr].to_numpy()
X_cont_wo = df[cont_attr].to_numpy()

# Removing outliers from selected variables (based on box plots from data_visualization)
for col in sel_attr:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1    # IQR is interquartile range.
    _filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    df = df.loc[_filter]

X = df[:-1].to_numpy()
X_cont = df[cont_attr].to_numpy()
X_sel = df[sel_attr].to_numpy()
y = df[y_attr].to_numpy()
attr = list(df[:-1].columns)

# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classNames = ["Less Chance for Heart Attack", "More Chance for Heart Attack"]
classDict = dict(zip(classNames, range(len(classNames))))

# no. of objects and attributes
N, M = X.shape
N_cont, M_cont = X_cont.shape
N_sel, M_sel = X_sel.shape
# no of classes
C = len(classNames)

##################################Part B

enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope and target
X_discrete = np.stack((X_sel[:, 1], X_sel[:, 2], X_sel[:, 7], y), axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
y = X_cont[:,3].T
y_label = "thalach"
X_cont = np.delete(X_cont, 3, axis=1)
X = np.column_stack((X_cont, X_enc))
# Add offset attribute
X_lr = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = X_lr.shape
attributeNames_lr = np.array(['Offset', 'age', 'trestbps', 'chol', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')

K = 10

# Values of lambda
lambdas = np.logspace(0, 5, 50)

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
for train_index, test_index in model_selection.KFold(K, shuffle=True, random_state = 0).split(X_lr, y):

    # extract training and test set for current CV fold
    X_train = X_lr[train_index]
    y_train = y[train_index]
    X_test = X_lr[test_index]
    y_test = y[test_index]
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
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title('Optimal lambda: 1e{0}'.format(np.round(np.log10(opt_lambda), 2)))
        plt.loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error', 'Validation error'])
        plt.grid()

    k += 1

min_error = np.min(Error_test_rlr)
opt_idx = np.argmin(Error_test_rlr)
best_weights = w_rlr[opt_idx,:]
best_lambda = lambdas_vect[opt_idx]


def best_regression_model(X_data):
    return X_data @ w_rlr[:, opt_idx]

plt.show()
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