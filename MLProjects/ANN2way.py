"""
ANN classifier - 2 lvl cross-fold validation
"""

import matplotlib.pyplot as plt
import numpy as np
# import torch
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import toolbox_02450
from toolbox_02450 import train_neural_net, draw_neural_net
from toolbox_02450 import rlr_validate
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


###  DATA PREPARATION #########################################################
# Removing thal from dataset (as discussed in previous report)
attributeNames = np.delete(sel_attr, -1, axis=0)
# ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']

X = np.delete(X_sel, -1, axis=1)

# One out of hot encoding 
enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope
X_discrete = np.stack((X_sel[:,1],X_sel[:,2],X_sel[:,7]),axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
X = np.column_stack((X_sel[:,0],X_sel[:,3:7],X_enc))
#update attributeNames
attributeNames = ['age','trestbps', 'chol', 'thalach', 'oldpeak',
       'female','male','typ_cp','atyp_cp',
       'no_cp','asymp_cp','slope_up','slope_flat','slope_down']
N, M = X.shape
y_m = np.asmatrix(y)
y = y_m.reshape((282,1))
C = 2

# Normalize data
X = stats.zscore(X);
K=10 

h_interval = np.array([1,2,3,4,5]) #here is where we define the array of hidden units
max_iter = 10000 
n_replicates = 1 #due to computation time

errors = np.empty((K,1)) # a list for storing generalizaition error after each outer cv-fold
h_optimal_list = [] # a list for storing optimal hidden units no after each outer cv-fold
ANN_best_models = []  # a list for models for storing models after each outer cv-fold

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


for k, (train_index, test_index) in enumerate(model_selection.StratifiedKFold(K, shuffle=True, random_state = 0).split(X,y)):
    print('\nCROSSVALIDATION OUTER FOLD: {0}/{1}'.format(k+1,K)) 
    
    #network_validate_classification does the inner cross-validation with cvf=10
    opt_val_err, opt_n_h_units = network_validate_classification(X,y,h_interval)
    
    model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, opt_n_h_units), #M features to H hiden units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(opt_n_h_units, 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid() # final tranfer function
                    )
    loss_fn = torch.nn.BCELoss()

    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    #y predicted
    y_sigmoid = net(X_test)
    y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8) #set tershold to classify as 0 or 1

    # Determine errors and errors
    y_test = y_test.type(dtype=torch.uint8)

    e = y_test_est != y_test
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    errors[k] = error_rate # store error rate for current CV fold 
    h_optimal_list.append(opt_n_h_units)
    ANN_best_models.append(net)
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

opt_index_2 = np.argmin(errors)
opt_val_err = np.min(errors)
opt_n_h_units = h_optimal_list[opt_index_2]
best_ANN_model = ANN_best_models[opt_index_2]

# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('Error rate');
summaries_axes[1].set_title('Test misclassification rates')

print('Diagram of best neural net:')
weights = [best_ANN_model[i].weight.data.numpy().T for i in [0,2]]
biases = [best_ANN_model[i].bias.data.numpy() for i in [0,2]]
tf =  [str(best_ANN_model[i]) for i in [1,3]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

print("Test errors are: ",errors)
print("Optimal hidden units for each fold: ",h_optimal_list)
print("Optimal test error is {}% with no of hidden units {}".format(np.round(opt_val_err*100,2),opt_n_h_units)) 
