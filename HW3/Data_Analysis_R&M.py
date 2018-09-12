__author__ = 'Skip To My Lou'

import pandas as pd
import numpy as np
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
from random import uniform
from math import sqrt

# Listing 2-1: Sizing Up a New Data Set
print("\nListing 2-1: Sizing Up a New Data Set")

data = pd.read_csv('sonar.all-data.csv', header=None)
print("Number of Rows of Data = " + str(len(data)))
print("Number of Columns of Data = " + str(len(data.iloc[0])))

# Listing 2-2: Determining the Nature of Attributes
print("\nListing 2-2: Determining the Nature of Attributes")

print(data.dtypes)

# Listing 2-3: Summary Statistics for Numeric and Categorical Attributes
print("\nListing 2-3: Summary Statistics for Numeric and Categorical Attributes")

def generate_summary4column(data, col=0, num_of_q_boundaries=4):
    col_data = data[col]
    print("Mean = ", np.mean(col_data))
    print("Standard Deviation = ", np.std(col_data))
    q_boundaries = []
    for i in range(0, num_of_q_boundaries):
        q_boundaries.append(np.percentile(col_data, i*(100) / num_of_q_boundaries))
    print("\nBoundaries for", num_of_q_boundaries, "Equal Percentiles\n", q_boundaries)
    print("\nCount for Each Value of Categorical Label\n", data.groupby([60])[col].count())

generate_summary4column(data, 3, 10)

# Listing 2-4: Quantile-Quantile Plot for 4th Rocks versus Mines Attribute
print("\nListing 2-4: Quantile-Quantile Plot for 4th Rocks versus Mines Attribute")

def Q_Q_plot4column(data, col=0, dist="norm", plot=pylab):
    stats.probplot(data[col], dist=dist, plot=plot)
    pylab.show()

Q_Q_plot4column(data, 4)

# Listing 2-5: Using Python Pandas to Read and Summarize Data
print("\nListing 2-5: Using Python Pandas to Read and Summarize Data")

print("Head of Data\n", data.head())
print("Tail of Data\n", data.tail())
print("Description of Data\n", data.describe())

# Listing 2-6: Parallel Coordinates Graph for Real Attribute Visualization
print("\nListing 2-6: Parallel Coordinates Graph for Real Attribute Visualization")

for i in range(0, len(data)):
    pcolor = "red" if data.iloc[i][60] == "M" else "blue"
    row_data = data.iloc[i, :60]
    row_data.plot(color = pcolor, alpha = 0.5)
plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

# Listing 2-7: Cross Plotting Pairs of Attributes
print("\nListing 2-7: Cross Plotting Pairs of Attributes")

def cross_plot_pairs(data, row1, row2):
    row1_data = data.iloc[row1, :60]
    row2_data = data.iloc[row2, :60]
    plt.scatter(row1_data, row2_data)
    plt.xlabel("No."+str(row1+1)+" Attribute")
    plt.ylabel("No."+str(row2+1)+" Attribute")
    plt.show()

cross_plot_pairs(data, 1, 2)
cross_plot_pairs(data, 1, 20)

# Listing 2-8: Correlation between Classification Target and Real Attributes
print("\nListing 2-8: Correlation between Classification Target and Real Attributes")

def correlation_plot(data, col, enhance=False):
    target = []
    for i in range(0, len(data)):
        target.append((1.0 + uniform(-0.1, 0.1) if enhance else 1.0) if data.iloc[i][60] == "M" else (0.0 + uniform(-0.1, 0.1) if enhance else 0.0))
    col_data = data[col]
    plt.scatter(col_data, target, alpha = 0.5 if enhance else None, s = 120 if enhance else None)
    plt.xlabel("Attribute Value")
    plt.ylabel("Target Value")
    plt.show()

correlation_plot(data, 35)
correlation_plot(data, 35, 1)

# Listing 2-9: Pearson's Correlation Calculation for Attributes
print("\nListing 2-9: Pearson's Correlation Calculation for Attributes")

def calculate_pearsons_correlation(data, row1, row2):
    row1_data = data.iloc[row1, :60]
    row2_data = data.iloc[row2, :60]
    mean1 = 0; mean2 = 0
    data_len = len(row1_data)
    for i in range(0, data_len):
        mean1 += row1_data[i] / data_len
        mean2 += row2_data[i] / data_len
    var1 = 0; var2 = 0
    for i in range(0, data_len):
        var1 += (row1_data[i] - mean1) ** 2 / data_len
        var2 += (row2_data[i] - mean2) ** 2 / data_len
    cor12 = 0
    for i in range(0, data_len):
        cor12 += (row1_data[i] - mean1) * (row2_data[i] - mean2) / (sqrt(var1 * var2) * data_len)
    print("Correlation between attribute " + str(row1+1) + " and " + str(row2+1) + "\n" +str(cor12))

calculate_pearsons_correlation(data, 1, 20)

# Listing 2-10: Presenting Attribute Correlations Visually
print("\nListing 2-10: Presenting Attribute Correlations Visually")

corMat = pd.DataFrame(data.corr())
plt.pcolor(corMat)
plt.show()

print("My name is Yijun Lou\n"
          "My NetId is: ylou4\n"
          "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")