#!/usr/bin/env python
# coding: utf-8

# ## Görev 1:  

# In[44]:


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

new_df = ["NUM_" + i.upper()  if df[i].dtype != "O" else i.upper() for i in df.columns]
new_df


# ## Görev 2: 

# In[51]:


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

new_df = ["NUM_" + i.upper()  if df[i].dtype != "O" else i.upper() for i in df.columns]
new_df

[ l + "_FLAG" if "NO" not in l else l for l in new_df]


# ## Görev 3: 

# In[87]:


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"]
 
df[num_cols].head()


# In[ ]:





# In[ ]:




