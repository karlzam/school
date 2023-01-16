
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Q1
# import the data
inp_red = np.load('inp_redshift.npy')

# plotting a histogram of each rows 1 index (second) entry for some reason:
# is it just as an example of one of the variables?
plt.figure(1)
plt.title("Normalized vs Un-Normalized Data")
plt.hist(inp_red[:,1])

# print the shape of the array
print(np.shape(inp_red))

# Next three steps are essentially one because they're done in this order
# This estimator scales and translates each feature individually such that it is in the given range on the training set,
# e.g. between zero and one.
# Use standard scalar when things follow a normal distribution
# Use min/max when well known over the domain, not necessarily a normal distribution
sc = MinMaxScaler()

# train the scaling function on the input redshift data. The fit method is calculating the mean and variance
# of each of the features present in our data.
sc.fit(inp_red)

# The transform method is transforming all the features using the respective mean and variance.
inp_red_norm = sc.transform(inp_red)

#numpy is row, column, plotting the second column of the normalized histo? not sure why? gunna swap it to 1
plt.hist(inp_red_norm[:,1])
plt.ylabel('Count')
plt.xlabel('2nd Variable in Dataset Magnitude')
plt.tight_layout()
plt.show()

# print the number of datapoints
np.shape(inp_red_norm)

# call the sklearn PCA decomposition
pca = PCA()

# Determine transformed features
inp_red_pca = pca.fit_transform(inp_red_norm)

# Determine explained variance using explained_variance_ration_ attribute
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# Create the visualization plot
plt.figure(2)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

print('test')