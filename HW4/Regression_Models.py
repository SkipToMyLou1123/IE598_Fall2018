import numpy as np
from sklearn.datasets.samples_generator import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from yellowbrick.regressor import ResidualsPlot

X, y, cf= make_regression(n_samples=1000, n_features=100, noise=0, coef=True, random_state=42)
np.savetxt("make_regressionX.csv", X, delimiter=",")
np.savetxt("make_regressionY.csv", y, delimiter=",")
np.savetxt("make_regressionCF.csv", cf, delimiter=',')

X = pd.read_csv('make_regressionX.csv', header=None)
y = pd.read_csv('make_regressionY.csv', header=None, prefix="L")

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

def summary4reg(reg, show=True):
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    # Accurate Score
    acc_score = reg.score(X_test, y_test)
    # MSE
    mse = mean_squared_error(y_test, y_pred)
    # R2
    r2 = r2_score(y_test, y_pred)
    coef = reg.coef_[0] if len(reg.coef_) == 1 else reg.coef_
    if show:
        print("Coefficient with Meaning")
        for i in range(0, len(coef)):
            if coef[i] > 0.00001:
                print("w" + str(i), coef[i])
        # uncomment this part for Linear Regression and Ridge Regression
        # visualizer = ResidualsPlot(reg, hist=False)
        # visualizer.fit(X_train, y_train)
        # visualizer.score(X_test, y_test)
        # visualizer.poof()
        # plt.show()
        # --------------------------------------------------
        # uncomment this part for Lasso Regression and ElasticNet Regression
        plt.scatter(y_pred, y_pred - y_test[0])
        plt.xlabel("Predicted value of y")
        plt.ylabel("Residuals")
        plt.show()
        # --------------------------------------------------
        print("Accurate Score: ", acc_score)
        print("MSE: ", mse)
        print("R2: ", r2)
    return acc_score, mse, r2

def test4alpha(reg, alpha_div = 50, alpha_min = -5, alpha_max = -2, compare_flag=1, isElasticNet = False):
    best_alpha = [0, 0, 0, 0]
    alpha_vals = np.logspace(alpha_min, alpha_max, alpha_div)
    if isElasticNet:
        best_l1 = 0
        l1_vals = np.linspace(0, 1, 10)
        for l1_r in l1_vals:
            reg.l1_ratio = l1_r
            for alpha in alpha_vals:
                reg.alpha = alpha
                acc, mse, r2 = summary4reg(reg, show=False)
                if acc > best_alpha[1] and compare_flag == 1 or (mse < best_alpha[2] and compare_flag == 2) or (
                        r2 > best_alpha[3] and compare_flag == 3):
                    best_alpha[0] = reg.alpha
                    best_alpha[1] = acc
                    best_alpha[2] = mse
                    best_alpha[3] = r2
                    best_l1 = l1_r
        print("Best l1_ratio of ElasticNet regression is: ", best_l1)
    else:
        for alpha in alpha_vals:
            reg.alpha = alpha
            acc, mse, r2 = summary4reg(reg, show=False)
            if acc > best_alpha[1] and compare_flag == 1 or (mse < best_alpha[2] and compare_flag == 2) or (r2 > best_alpha[3] and compare_flag == 3):
                best_alpha[0] = reg.alpha
                best_alpha[1] = acc
                best_alpha[2] = mse
                best_alpha[3] = r2

    reg.alpha = best_alpha[0]
    summary4reg(reg)
    print("Best alpha of current regression is: ", best_alpha[0])

# Part1 Exploratory Data Analysis
d = pd.concat([X, y], axis=1)
corMat = pd.DataFrame(d.corr())
max_col = 0; max_row = 0; cor_val = 0
cor4y = corMat.iloc[len(corMat)-1][:-1]
mean4cor4y = np.mean(cor4y)
print("Mean of correlation of each attributes to y:", mean4cor4y)
std4cor4y = np.std(cor4y)
print("Standard deviation of this series: ", std4cor4y)
col_list = []
for i in range(0, len(cor4y)):
    if corMat.iloc[len(corMat)-1][i] > mean4cor4y + 1.5 * std4cor4y:
        col_list.append(i)
print("Meaningful attribute column index:", col_list)
new_d = pd.concat([d[col_list],y], axis=1)
sns.pairplot(new_d, plot_kws={"s": 3})
plt.show()
sns.heatmap(new_d.corr(), annot=True)
plt.show()

# Part2 Linear Regression
print("Linear Regression Summury")
reg = linear_model.LinearRegression()
summary4reg(reg)
print("-"*60)

# Part3.1 Ridge Regression
print("Ridge Regression Summury")
reg_r = linear_model.Ridge()
test4alpha(reg_r)
print("-"*60)

# Part3.2 Lasso Regression
print("Lasso Regression Summury")
reg_l = linear_model.Lasso()
test4alpha(reg_l)
print("-"*60)

# Part3.3 ElasticNet Regression
print("ElasticNet Regression Summury")
reg_e = linear_model.ElasticNet()
test4alpha(reg_e, isElasticNet=True)
print("-"*60)

print("My name is Yijun Lou\n"
              "My NetId is: ylou4\n"
              "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

