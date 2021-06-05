# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Random seed
SEED = np.random.seed(2020)
print("\n...... Processing Data ......\n")
io ='/home/lilabguest2/shenhaoyu/m6A/Mammalian.xlsx'
data = pd.read_excel(io, sheet_name = 2)
data.rename(columns={'Sequence context':'cdna','Label':'tag'},inplace=True) # 修改列名 
for i in range(len(data)):
    l=list(data.iloc[i,0])
    for j in range(len(l)):
        if l[j]=='-':
            l[j]=''
    s=''.join(l)
    data.iloc[i,0]=s
data_pos=data[data['tag']==1]
data_neg=data[data['tag']==0]
data_neg.to_csv('/home/lilabguest2/shenhaoyu/negative_m6A.csv', sep=',',na_rep='NA', header=True,index=False)
data_pos.to_csv('/home/lilabguest2/shenhaoyu/positive_m6A.csv', sep=',',na_rep='NA', header=True,index=False)