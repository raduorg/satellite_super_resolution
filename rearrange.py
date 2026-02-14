PATH = '/home/raduorg/Master/DeepLearning/kaggle2/hyperparam_results/mlp_results_20260211_221601.csv'

import pandas as pd

df = pd.read_csv(PATH)
df = df.sort_values(by=['psnr'], ascending=False)
print(df)
df.to_csv('hyperparam_results/mlp_results_20260211_221601_sorted.csv', index=False)