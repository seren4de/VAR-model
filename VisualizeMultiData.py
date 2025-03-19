import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# calling IPython API to run line magic from the script 
#from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

#'exec(%matplotlib inline)'
#%matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
df.to_csv('data.csv', header=None, index=None, sep=',', mode='a')
print(df.shape)  # (123, 8)
df.tail()

# Plot
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();
plt.show()

