#!/usr/bin/env python
# coding: utf-8

# ## Soru 1:  persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# Load the CSV file into a DataFrame
df = pd.read_csv("persona.csv")
df.head()


# In[25]:


df.describe().T


# In[26]:


def check_df(dataframe, head=5):
    print("############ COLUMNS############")
    print(df.columns)
    print("############ SHAPE #############")
    print(dataframe.shape)
    print("\n############ TYPE #############")
    print(dataframe.dtypes)
    print("\n############ INFO #############")
    print(df.info())
    print("\n############ HEAD #############")
    print(dataframe.head())
    print("\n############ TAİL #############")
    print(dataframe.tail())
    print("\n############ EKSIK VERİ SAYISI #############")
    print(dataframe.isnull().sum())
    print("\n############ QUANTİLE #############")
    ## Sayısal değişkenlerin dağılım bilgisi, %0, %5,%50 , %95, %100
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 1]))
df = pd.read_csv("persona.csv")

## yukarıdaki check_df(df)  fonksiyonu başka bir veri setinede uygularız
check_df(df)


# ## Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

# In[27]:


#  the 'SOURCE' column'da  unique değer sayısı 
unique_sources = df['SOURCE'].unique()

# value_counts  yani frekans sayısı
source_frequencies = df['SOURCE'].value_counts()

print(f"Unique SOURCE değeri: {unique_sources}")
print("\n Her SOURCE için frekans değerleri:")
print(source_frequencies)


# ## Soru 3:   Kaç unique PRICE vardır?

# In[28]:


unique_price = df["PRICE"].unique()
unique_price


# ## Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

# In[29]:


# PRICE'a göre satış frekanslarını hesapla
price_frequencies = df["PRICE"].value_counts()

price_frequencies


# ## Soru 5: Hangi ülkeden kaçar tane satış olmuş?

# In[55]:


df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()


# ## Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

# In[31]:


country_revenue = df.groupby("COUNTRY")["PRICE"].sum()
country_revenue


# ##  Soru 7: SOURCE türlerine göre satış sayıları nedir?

# In[56]:


df["SOURCE"].value_counts()


# ## Soru 8: Ülkelere göre PRICE ortalamaları nedir?

# In[57]:


country_revenue_mean = df.groupby("COUNTRY")["PRICE"].mean()
country_revenue_mean


# ## Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

# In[61]:


source_revenue_mean = df.groupby("SOURCE")["PRICE"].mean()
source_revenue_mean


# ## Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

# In[35]:


country_source_revenue_mean = df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()
country_source_revenue_mean


# ## Görev 2:    COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

# In[36]:


revenue_average = df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE": "mean"})
revenue_average


# ## Görev 3:  Çıktıyı PRICE’a göre sıralayınız.

# In[37]:


agg_df  = revenue_average.sort_values(by ="PRICE", ascending=False)
agg_df.head(20)


# ## Görev 4:  Indekste yer alan isimleri değişken ismine çeviriniz.

# In[38]:


agg_df = agg_df.reset_index()

agg_df.head()


# ## Görev 5:  Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

# In[68]:


bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i bölelim:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()


# ## Görev 6:  Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

# In[71]:


agg_df['customers_level_based'] = agg_df.apply(
    lambda row: f"{row['COUNTRY'].upper()}_{row['SOURCE'].upper()}_{row['SEX'].upper()}_{row['AGE_CAT'].upper()}", 
    axis=1
)
result_df = agg_df[['customers_level_based', 'PRICE']]
result_df


# In[72]:


# Tekrar eden customers_level_based değerlerini bulma
value_counts = result_df['customers_level_based'].value_counts()
value_counts


# In[74]:


agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df 


# In[77]:


agg_df = agg_df.reset_index()
agg_df


# In[78]:


agg_df["customers_level_based"].value_counts()
agg_df.head()


# ## Görev 7:  Yeni müşterileri (personaları) segmentlere ayırınız.

# In[80]:


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4 ,labels=["D","C","B","A"])
agg_df


# ##### Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

# In[47]:


agg_df_group = agg_df.groupby(["SEGMENT"]).agg({"PRICE": ("mean")})
                                                            
agg_df_group


# ## Görev 8:  Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.

# In[81]:


new_user1 = "TUR_ANDROID_FEMALE_31_40"
new_user2 = "FRA_IOS_FEMALE_31_40"
# Ortalama gelir hesaplama

segment_df = agg_df[(agg_df["customers_level_based"] == new_user1) | (agg_df["customers_level_based"] == new_user2) ]
average_revenue = segment_df["PRICE"].mean()
segment_df 


# In[ ]:





# In[ ]:




