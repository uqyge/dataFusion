#%%
import pandas as pd
import numpy as np
import plotly.express as px
# %%
df = pd.read_excel("./data/2012005ROWS.xlsx")
df.head()

# %%
x_size = 20
y_size = 120
t_step = 61
# 20 * 120 * 61
df.iloc[x_size * y_size]

#%%
spliter_loc=[v for t in range(1,t_step) for v in (t*x_size*y_size, t*x_size*y_size+1)]

len(spliter_loc)

#%%
df_new = df.drop(spliter_loc)
df_new.shape

#%%
t = np.asarray([[i]*x_size*y_size for i in range(t_step)])
df_new['time']=t.flatten()
#%%
df_new.head()

df_new.to_parquet('wudi.parquet')
#%%
