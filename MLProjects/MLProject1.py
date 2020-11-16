# 02450: Introduction to Machine Learning & Data Mining
# Project 1 - Data: Feature extraction, and visualization
# Adrienne Cohrt s184426 & Viktor Tsanev s184453

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA

data = pd.read_csv('characters.csv')
print(data)

# Iterate through the table and give number of missing values for each attribute
columns = list(data)
for col in columns:
    empty = 0
    for row in range(87):
        if str(data[col][row]) == "nan":
            empty += 1
    print("Attribute " + col + " has " + str(empty) + " missing values.")

###############Plots#################
# Box plot - outliers height, mass & human height
fig1, ax1 = plt.subplots()
height = []
ax1.set_title("Height Box Plot")
for i in range(87):
    if str(data["height"][i]) != "nan":
        height.append(data["height"][i])
ax1.boxplot(height)

fig2, ax2 = plt.subplots()
mass = []
ax2.set_title("Mass Box Plot")
for j in range(87):
    if str(data["mass"][j]).isdigit():
        mass.append(float(data["mass"][j]))
ax2.boxplot(mass)

fig3, ax3 = plt.subplots()
heightHuman = []
ax3.set_title("Human Height Box Plot")
for i in range(87):
    if str(data["height"][i]) != "nan" and str(data["species"][i]).lower() == "human":
        heightHuman.append(data["height"][i])
ax3.boxplot(heightHuman)

fig4, ax4 = plt.subplots()
scatterH = []
scatterM = []
ax4.set_title("Height vs. Mass")
for i in range(87):
    if str(data["height"][i]) != "nan" and str(data["mass"][i]).isdigit():
        scatterH.append(data["height"][i])
        scatterM.append(data["mass"][i])
scatterM = [int(i) for i in scatterM]
ax4.scatter(scatterH, scatterM)

pca = PCA().fit(np.column_stack((scatterH, scatterM)))
fig5, ax5 = plt.subplots()
ax5.plot(np.cumsum(pca.explained_variance_ratio_))
ax5.set_xlabel('number of components')
ax5.set_ylabel('cumulative explained variance')
#################Summary Statistics#################################
print("Height ", min(height), "to", max(height))
print("Mean ", np.mean(height), " SD", np.std(height), " Median", np.median(height))

mass2 = np.append(mass, 1358)
print("Mass ", min(mass2), "to", max(mass2))
print("Mean", np.mean(mass2), " SD", np.std(mass2), " Median", np.median(mass2))

humans = 0
non = []
for i in range(87):
    if str(data["species"][i]) == "Human":
        humans = humans +1
    else:
        non.append(data["species"][i])
nonH = len(set(non))
print("Number of humans: ", humans)
print("Number of other species: ", nonH)

plt.show()