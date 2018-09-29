import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
print(dataset.describe())

# Plot
sns.pairplot(dataset.iloc[:, 0:13])
sns.heatmap(dataset.iloc[:, 0:13].corr(), annot=True)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

def apply_to_classifier(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, method = "pca", classifier = "logreg", standardize = True, transform = True, plot = False, gamma = 15):

    if standardize:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    if transform:
        principle = PCA(n_components=2) if method is "pca" else (LDA(n_components=2) if method is "lda" else KernelPCA(n_components=2, kernel='rbf', gamma=gamma))
        X_train = principle.fit_transform(X_train) if method is "pca" else principle.fit_transform(X_train, y_train)
        X_test = principle.transform(X_test)

    classifier = LogisticRegression(random_state=42) if classifier is 'logreg' else SVC(C=0.5, kernel='rbf', random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    c_mat = confusion_matrix(y_test, y_pred)

    # accuracy scores
    train_acc_score = classifier.score(X_train, y_train)
    test_acc_score = classifier.score(X_test, y_test)
    print("Accuracy scores for both in sample and out of sample are: ", train_acc_score, test_acc_score)
    print(c_mat)

    # plot
    if plot:
        plot_decision_regions(X_train, y_train, clf=classifier)
        plt.show()

# Part 2: Logistic regression classifier v. SVM classifier - baseline
apply_to_classifier(transform=False, classifier='logreg')
apply_to_classifier(transform=False, classifier='svm')

# Part 3: Perform a PCA on both datasets
apply_to_classifier(classifier='logreg', method='pca')
apply_to_classifier(classifier='svm', method='pca')

# Part 4: Perform and LDA on both datasets
apply_to_classifier(classifier='logreg', method='lda')
apply_to_classifier(classifier='svm', method='lda')

# Part 5: Perform a kPCA on both datasets
apply_to_classifier(classifier='logreg', method='kPCA')
apply_to_classifier(classifier='svm', method='kPCA')
for i in np.linspace(0, 10, 40):
    print(i)
    if i == 0:
        continue
    apply_to_classifier(classifier='logreg', method='kPCA',gamma = i)
    apply_to_classifier(classifier='svm', method='kPCA', gamma = i)

print("My name is Yijun Lou\n"
              "My NetId is: ylou4\n"
              "I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

