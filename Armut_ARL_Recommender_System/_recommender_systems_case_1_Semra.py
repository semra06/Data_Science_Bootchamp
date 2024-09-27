#!/usr/bin/env python
# coding: utf-8

# ## Task 1: Preparing Data

# In[ ]:


## Armut, Turkey's largest online service platform, brings together those who provide 
## services and those who want to receive services.
## It is desired to create a product recommendation system with Association Rule Learning
## using the data set containing the users who receive the service and the services and 
## categories that these users receive.


# In[ ]:


## Step 1:  armut_data.csv read csv file
import pandas as pd
import datetime as dt
from mlxtend.frequent_patterns import apriori, association_rules
df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.head()


# In[12]:


df.describe().T


# In[13]:


df.isnull().sum()


# In[14]:


df.shape


# In[50]:


## Step 2: ServiceID represents a different service for each CategoryID. 
## Create a new variable to represent these services by combining ServiceID and CategoryID with "_".
## The axis=1 parameter causes the apply function to work on a row-by-row basis.
df["Hizmet"] =df.apply(lambda x: str(x["ServiceId"]) + "_" + str(x["CategoryId"]), axis=1)
df.head()


# In[51]:


## Step 3: The data set consists of the date and time the services were received, 
## there is no basket definition (invoice etc.). In order to apply ASL, a basket (invoice etc.) definition must be created. 
## Here, the basket definition is the services that each customer receives monthly.
## Baskets must be identified with a unique ID.
## Convert CreateDate column to datetime format
df['CreateDate'] = pd.to_datetime(df['CreateDate'])

## only takes month and year information, (YYYY-MM)
df['New_Date'] = df['CreateDate'].dt.to_period('M').astype(str)

## in order to apply ASL create unique SepetID
df["SepetID"] = df.apply(lambda x: str(x["UserId"]) + "_" + str(x["New_Date"]), axis=1)

df.head()


# In[52]:


df.shape


# ## Task 2: Generate Association Rules

# In[53]:


## ARL Preparing the Data Structure (SepetID-Product Matrix)- That is, "SepetID" in the rows and "Hizmet" in the columns.
## Creating service pivot table, pivot this by saying Unstack
pivot_df = df.groupby(['SepetID', 'Hizmet'])["Hizmet"].count().unstack().fillna(0)
pivot_df = pivot_df.applymap(lambda x: 1 if x > 0 else 0)
pivot_df.head()


# In[74]:


## Step 2: Create association rules with apriori
frequent_itemsets = apriori(pivot_df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


# In[76]:


## Use the arl_recommender function to recommend a service to a user who last received the 2_0 service.
sorted_rules = rules.sort_values("support", ascending=False)
product_id = "2_0"
recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
print(recommendation_list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




