#!/usr/bin/env python
# coding: utf-8

# ## Görev 1: 

# In[72]:


import seaborn as sns
df = sns.load_dataset("titanic")
df.head()


# ## Görev 2: 

# In[73]:


df["sex"].value_counts()


# In[74]:


df["sex"].count()


# ## Görev 3: 

# In[75]:


df.nunique()


# In[76]:


df.apply(lambda i: i.nunique())


# ## Görev 4: 

# In[77]:


pclass_unique_count = df["pclass"].unique()
print(f"pclass değişkeninin unique değerlerin{pclass_unique_count}")
df[["pclass"]].apply(lambda x: x.unique())


# ## Görev 5:  

# In[78]:


df[["pclass","parch"]].nunique()


# In[79]:


unique_counts = df[['pclass', 'parch']].apply(lambda x: x.unique())
print(unique_counts)


# ## Görev 6:  

# In[80]:


df["embarked"].dtype


# In[81]:


df["embarked"] = df["embarked"].astype("category")
df.dtypes


# ## Görev 7:  

# In[82]:


## filtreleme
df[df["embarked"]=="C"].head(10)


# ## Görev 8:  

# In[83]:


df[df["embarked"] != "S"].head(10)


# ## Görev 9:  

# In[84]:


df[(df["age"] < 30)  &  (df["sex"] == "female")].head()


# ## Görev 10:  

# In[85]:


df[(df["fare"] > 500 ) | (df["age"] > 70) ] 


# ## Görev 11: 

# In[86]:


df.isnull().sum()


# ## Görev 12:

# In[87]:


df_new = df.drop("who", axis=1)
df_new


# ## Görev 13:

# In[88]:


print("Eksik değerler öncesi:\n", df["deck"].isnull().sum())


# In[89]:


## mode en çok tekrar eden sayı bulur
mode_deck = df["deck"].mode()
mode_deck


# In[90]:


# Boş değerleri mod değeri ile doldurma
df["deck"].fillna(mode_deck, inplace=True)
df


# ## Görev 14:

# In[91]:


print(" age değikenindeki boş değerler sayısı: \n",df["age"].isnull().sum())
median_age = df["age"].median()
print("age'in median değeri: \n",median_age )
print("age değişkenindeki boş değerleri age değişkenin medyanı ile doldurunuz.:")
df["age"].fillna(median_age, inplace=True)
df


# ## Görev 15:  

# In[92]:


df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})


# ## Görev 16:  

# In[93]:


def age_flag(l):
    if l < 30:
        return 1
    else:
        return 0
df["age_flag"] = df["age"].apply(lambda x :  age_flag(x))
df.head()


# In[94]:


import seaborn as sns
df = sns.load_dataset("titanic")
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0) 
df.head()


# ## Görev 17: 

# In[95]:


import seaborn as sns
df = sns.load_dataset("Tips")
df.head()


# In[96]:


df.groupby("time").agg({"total_bill" : ["sum", "min", "max"]})


# ## Görev 19: 

# In[97]:


df.groupby(["time", "day"]).agg({"total_bill" : ["sum", "min", "max", "mean"]})


# ## Görev 20:  

# In[98]:


# Lunch zamanına ve kadın müşterilere ait veriyi filtreleyelim
filtered_df = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]
filtered_df.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                  "tip": ["sum", "min", "max", "mean"]})


# ## Görev 21: 

# In[99]:


df_filter = df.loc[(df["size"] < 3) & (df["total_bill"] > 10)].head()
df_filter  

average_total_bill = df_filter ['total_bill'].mean()

average_total_bill


# ## Görev 22: 

# In[100]:


df["total_bill_tip_sum"] =  df["total_bill"] + df["tip"]
df.head()


# ## Görev 23:  

# In[102]:


df_sorted = df.sort_values(by = "total_bill_tip_sum", ascending=False, ignore_index=True)[:30]
df_sorted_30


# In[ ]:




