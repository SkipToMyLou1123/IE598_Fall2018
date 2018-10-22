
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_EC = pd.read_csv('/Users/danaoqueyang/Desktop/MLF_GP2_EconCycle.csv')


## Part1: EDA

## Scatter plot
cols = ['T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index',
        'T10Y Index','CP1M','CP3M','CP6M']

sns.pairplot(df_EC[cols], size = 2)
plt.tight_layout()
plt.savefig('ScatterplotMatrix.png')
plt.show()


## Create heatmap
cm = np.corrcoef(df_EC[cols].values.T)

sns.set(font_scale=0.5)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size':6.5},yticklabels=cols,xticklabels=cols)
plt.savefig('heatmap.pdf')
plt.show()