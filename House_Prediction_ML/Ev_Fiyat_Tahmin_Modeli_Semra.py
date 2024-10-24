#!/usr/bin/env python
# coding: utf-8

# ## Görev 1: Keşifçi Veri Analizi

# In[340]:


# 1. GEREKLILIKLER

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier


# In[341]:


## Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.

# Train ve Test veri setlerini okuyoruz
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## Test veri setine bir 'SalePrice' sütunu ekleyelim ve bu sütunu NaN ile dolduralım
## Test veri setine, daha sonra tahmin edeceğimiz 'SalePrice' sütununu ekler ve eksik (NaN) olarak bırakır.
test['SalePrice'] = pd.NA

# Train ve Test veri setlerini birleştirme
## Train ve test veri setlerini birleştirir. axis=0 satırlar boyunca birleştirme işlemi yapar
## ignore_index=True ise eski indeksleri dikkate almadan yeni bir indeks oluşturur.
data = pd.concat([train, test], axis=0, ignore_index=True).reset_index()

# Birleştirilmiş veri setinin ilk birkaç satırına bakalım
data.head()


# In[342]:


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    # Sadece sayısal sütunları seçip quantile işlemi yapalım
    num_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    print(dataframe[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(data)


# In[343]:


data.shape


# In[344]:


# Her sütunda kaç tane eksik değer (NaN) olduğunu kontrol edelim
missing_values = data.isnull().sum()

# Sadece eksik değeri olan sütunları gösterelim
missing_values = missing_values[missing_values > 0]

# Eksik değer sayısını büyükten küçüğe sıralayalım
missing_values.sort_values(ascending=False, inplace=True)

# Eksik değerleri görelim
print(missing_values)


# In[345]:


# 'Id' sütununu veri setinden silme
data.drop(columns=['Id'], inplace=True)

# İlk birkaç satıra tekrar bakalım
print(data.head())


# In[346]:


## Adım 2:  Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'cat_but_car: {len(cat_but_car)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, num_cols, num_but_cat, cat_but_car

# df veri setimizi grab_col_names fonksiyonundan geçiriyoruz
cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(data)


# In[347]:


print(f"Kategorik Kolonlar: cat_cols: 52")
print("################################")
print(cat_cols)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Numerik Kolonlar: num_cols: 28")
print("################################")
print(num_cols)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Numerik Gözüken Kategorik Kolonlar: num_but_cat: 10")
print("################################")
print(num_but_cat)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Kategorik Gözüken Kardinal Kolonlar: cat_but_car: 1")
print("################################")
print(cat_but_car)
print("-------------------------------------------------------------------")
print("                                                                   ")


# In[348]:


## Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Veri setindeki her sütunun veri tipini inceleyelim

#Sayısal tip hataları: Örneğin, bazı sütunlar sayısal (int, float) olması gerekirken nesne (object) olarak gözükebilir.
#Kategorik tip hataları: Kategorik olması gereken sütunlar sayısal tipte olabilir.
print(data.dtypes)



# ### KATEGORİK DEĞİŞKENLERİN ANALİZİ

# In[349]:


## Öncelikle numerik ve kategorik değişkenlerin analizini yapılır.
## 1. olarak Kategorik değişkenlerin analizi yapılır 
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(data, col)


# ### NUMERİK DEĞİŞKENLERİN ANALİZİ

# In[350]:


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(data, col, plot=True)


# ### KATEGORİK DEĞİŞKENLERİN TARGET'A GÖRE ANALİZİ
# 

# In[351]:


# KATEGORİK DEĞİŞKENLERİN TARGET'A GÖRE ANALİZİ
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(data,"SalePrice",col)


# ##################################
# # KORELASYON
# ##################################

# In[352]:


# değişkenler arasındaki ilişkiyi incelemek için korelasyon analizine bakalım
# korelasyon iki değişken arasındaki ilişkinin yönünü ve derecesini gösterir
# -1 ile +1 arasında değişir ve 0 herhangi bir ilişki olmadığını gösterir
# -1 e yaklaştıkça negatif güçlü ilişki, +1 e yaklaştıkça pozitif güçlü ilişki olduğunu gösterir
corr = data[num_cols].corr()
corr


# In[353]:


# Korelasyonların gösterilmesi
# renk kırmızıya doğru kaydıkça negatif güçlü ilişki artmaktadır,
# renk koyu maviye doğru kaydıkça da pozitif güçlü ilişki artmaktadır
sns.set(rc={'figure.figsize': (15, 15)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


# In[354]:


## Korelasyon analizi yapacağız.

data[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(data[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte




# In[355]:


def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    # Select only numeric columns for correlation calculation
    numeric_df = dataframe.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    corr = numeric_df.corr()
    # Take the absolute value of the correlation matrix
    cor_matrix = corr.abs()
    # Create an upper triangle matrix to avoid duplicate checks
    # (Korelasyon matrisinin üst üçgeni oluşturulur, böylece aynı korelasyon çiftlerinin tekrar tekrar kontrol edilmesi önlenir.)
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    # Find columns with correlation above the threshold [Korelasyonu verilen eşikten (corr_th) yüksek olan sütunlar bulunur.]
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # Plot the heatmap if requested
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(data, plot=False)


# ## Görev 2: Feature Engineering

# In[356]:


# Eksik gözlemler sayısını kontrol etme
missing_values = data.isnull().sum()
print(missing_values)


# ### AYKIRI DEĞER ANALİZİ
# 
# 

# In[357]:


# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# baskılama aykırı değerlerin en alt değerlere ve en üst değerlere göre sabitlenmesi durumudur

# thresholdumuzu veri setimize göre gözlem sayısına göre, değişkenlerin yapısına göre kendi know-how ımıza göre belirleyebiliriz
# genel geçer %75 e %25 şeklinde alınandır. ancak çok fazla bilgi kaybetmemek için bu değerleri büyütmek mümkündür
# fazlaca baskılama yapmak çoğu zaman bilgi kaybına ve değişkenlerin arasındaki ilişkinin kaybolmasına neden olabilmektedir

# Sadece sayısal sütunlar üzerinde aykırı değer analizi yapalım
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    print(f"{col}: {outlier_thresholds(data, col)}")


# In[358]:


# Aykırı değer kontrolü

# bu eşik değerlerlere göre aykırı değerler var mı değişkenlerde, varsa hangilerinde var kontrol edeceğiz
# bir değişkenin aykırı değerlerini bool olarak sorgulatacağız
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(data, col))


# In[359]:


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(data,col)


# In[360]:


# tekrar bakalım aykırı değer kalmış mı
for col in num_cols:
    print(col, check_outlier(data, col))


# ### EKSİK DEĞER ANALİZİ

# In[361]:


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(data)
#Alley: Type of alley access ---- Sokak erişim türü


# In[362]:


# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir,
# bu kanıya data seti iyice inceleyerek ve data setin ve değişkenlerin dinamiklerine bakarak karar vermeliyiz
# örneğin PoolQC bir gözlemde boş ise o evde havuz olmadığını belirtmektedir.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]


# In[363]:


# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
# burada değişkenler kendi nezdinde incelemeli hepsine medyan ya da mod ya da ortalama uygulamak yerine
# değişken bazında uygun metrik ile doldurmak daha uygun olacaktır
for col in no_cols:
    data[col].fillna("No",inplace=True)

missing_values_table(data)


# In[364]:


# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(dataframe, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = dataframe[target]

    print("# BEFORE")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        dataframe = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    dataframe[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")

    return dataframe


data = quick_missing_imp(data, num_method="median", cat_length=17)


# In[365]:


## Kategorik için ise mod kullanılır.
## Sayısal verileri ortalama veya medyan ile doldur
## Eğer dağılım normal dağılımsa mean ile doldur, çarpık dağılımsa medyan
## Çarpık verileri histogramı çizdiriyoruz. Sağa sola çarpık ise median ile doldur. Değilse, mean ile doldur. 
## yani %50 den fazla ise, no yazacağız
## eğer azsa mode ile mi dolduruyoruz


# In[366]:


missing_values_table(data)
# ve hiç eksiklik kalmadı, SalePrice hariç


# ### RARE ANALİZİ YAPINIZ VE RARE ENCODER UYGULAYINIZ 

# In[367]:


# Kategorik kolonların dağılımının incelenmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        # Bu satır, her kategorik sütunun kaç farklı kategoriye sahip olduğunu yazdırır.
        print(col, ":", len(dataframe[col].value_counts()))

        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        # COUNT: Her kategorinin kaç kez tekrarlandığını (frekans) gösterir.
        # RATIO: Her kategorinin toplam veri kümesindeki oranını hesaplar.
        # TARGET_MEAN: Hedef değişkenin (SalePrice) her kategori için ortalama değerini hesaplar.
rare_analyser(data, "SalePrice", cat_cols)


# In[368]:


# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    # Orijinal veri kümesini değiştirmemek için bir kopyasını oluşturduk.

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    # Kategorik değişkenler (tür olarak O - object) arasında nadir kategorilere sahip olanları belirler.
    # Bir kategorinin nadir sayılması için frekans oranının rare_perc'den küçük olması gerekmektedir.

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        # Her nadir kategorik sütun için:
        # Kategorilerin frekans oranlarını hesaplar.
        # Nadir kategorileri belirler (frekans oranı rare_perc'den küçük olanlar).
        # Bu nadir kategorileri "Rare" etiketiyle değiştirir.

    return temp_df.head(100)


rare_encoder(data, 0.01)

# rare_encoder(df, 0.01) fonksiyonunu çağırdığınızda, df veri kümesindeki nadir kategoriler 0.01 (yani %1) frekans oranının altında olanlar olarak belirlenir ve
# bu kategoriler "Rare" etiketiyle değiştirilir.


# ## FEATURE ENGINEERING

# ### Yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz.

# In[369]:


data["NEW_1st*GrLiv"] = data["1stFlrSF"] * data["GrLivArea"]

data["NEW_Garage*GrLiv"] = (data["GarageArea"] * data["GrLivArea"])

# Total Floor
data["NEW_TotalFlrSF"] = data["1stFlrSF"] + data["2ndFlrSF"]

# Total Finished Basement Area
data["NEW_TotalBsmtFin"] = data.BsmtFinSF1 + data.BsmtFinSF2

# Porch Area
data["NEW_PorchArea"] = data.OpenPorchSF + data.EnclosedPorch + data.ScreenPorch + data["3SsnPorch"] + data.WoodDeckSF

# Total House Area
data["NEW_TotalHouseArea"] = data.NEW_TotalFlrSF + data.TotalBsmtSF

data["NEW_TotalSqFeet"] = data.GrLivArea + data.TotalBsmtSF


# Lot Ratio
data["NEW_LotRatio"] = data.GrLivArea / data.LotArea

data["NEW_RatioArea"] = data.NEW_TotalHouseArea / data.LotArea

data["NEW_GarageLotRatio"] = data.GarageArea / data.LotArea

# MasVnrArea
data["NEW_MasVnrRatio"] = data.MasVnrArea / data.NEW_TotalHouseArea

# Dif Area
data["NEW_DifArea"] = (data.LotArea - data["1stFlrSF"] - data.GarageArea - data.NEW_PorchArea - data.WoodDeckSF)


data["NEW_OverallGrade"] = data["OverallQual"] * data["OverallCond"]


data["NEW_Restoration"] = data.YearRemodAdd - data.YearBuilt

data["NEW_HouseAge"] = data.YrSold - data.YearBuilt

data["NEW_RestorationAge"] = data.YrSold - data.YearRemodAdd

data["NEW_GarageAge"] = data.GarageYrBlt - data.YearBuilt

data["NEW_GarageRestorationAge"] = np.abs(data.GarageYrBlt - data.YearRemodAdd)

data["NEW_GarageSold"] = data.YrSold - data.GarageYrBlt


# In[370]:


# Convert relevant columns to numeric, coercing errors to NaN
columns_to_sum = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                  "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", 
                  "GarageCond", "Fence"]

# Apply pd.to_numeric() to each column
data[columns_to_sum] = data[columns_to_sum].apply(pd.to_numeric, errors='coerce')

# Now perform the summation
data["TotalQual"] = data[columns_to_sum].sum(axis=1)


# In[371]:


# kolonlar üzerinden yeni feature lar ürettik ve eskilerine gerek kalmadı bu yüzden bunlara ihtiyacımız yok ve data frame den düşüreceğiz
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
data.drop(drop_list, axis=1, inplace=True)


# In[372]:


data.shape


# ### Label Encoding & One-Hot Encoding işlemlerini uygulayınız.

# In[373]:


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in data.columns if data[col].dtypes == "O" and len(data[col].unique()) == 2]

for col in binary_cols:
    label_encoder(data, col)


# In[374]:


data.head()


# In[375]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # Filter out columns that don't exist in the dataframe
    existing_cols = [col for col in categorical_cols if col in dataframe.columns]
    
    # Perform one-hot encoding only on existing columns
    dataframe = pd.get_dummies(dataframe, columns=existing_cols, drop_first=drop_first)
    
    # Fill NaN values with 0
    dataframe = dataframe.fillna(0)
    
    # Convert the DataFrame to integers
    dataframe = dataframe.astype(int)
    
    return dataframe

# Now apply the one-hot encoder
data = one_hot_encoder(data, cat_cols, drop_first=True)
data.head()


# ## MODELLEME

# In[376]:


#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_data = data[data['SalePrice'].notnull()]
test_data = data[data['SalePrice'].isnull()]


# In[377]:


# Train verisi ile model kurup, model başarısını değerlendiriniz.
# bağımlı ve bağımsız değişkenleri seçiyoruz
# daha sonra da log dönüşümü yaparak model kuracağız ve rmse değerlerimizi log öncesi ve log sonrasına göre karşılaştıracağız
y = train_data['SalePrice'] # np.log1p(df['SalePrice'])  y= bağımlı değişken
X = train_data.drop(["index", "SalePrice"], axis=1)        # X = Id hariç bağımsız değişkenler (90 değişkenle beraber)


# In[378]:


# Train verisi ile model kurup, model başarısını değerlendiriniz.
# modelimizi kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


# In[379]:


# kullanacağımız yöntemi import ettik
from lightgbm import LGBMRegressor
# bağımlı değişkenimiz sayısal ise regression, regressor algoritmalarını
# bağımlı değişkenimiz kategorikse classification algoritmalarını kullanıyoruz

# kullanacağımız yöntemleri içeren bir model tanımlı nesne kuruyoruz
# kapalı olan algoritmaları da açarak onları da modele sokabilirsiniz
models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


# In[380]:


# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# In[381]:


df['SalePrice'].mean()


# In[382]:


df['SalePrice'].std()


# ### BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.

# In[383]:


# Log dönüşümünün gerçekleştirilmesi

# tekrardan Train ve Test verisini ayırıyoruz.
train_data = data[data['SalePrice'].notnull()]
test_data = data[data['SalePrice'].isnull()]
# Bağımlı değişkeni normal dağılıma yaklaştırarak model kuracağız

y = np.log1p(train_data['SalePrice'])
X = train_data.drop(["index", "SalePrice"], axis=1)


# In[384]:


# Verinin eğitim ve test verisi olarak bölünmesi
# log dönüşümlü hali ile model kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


# In[385]:


# bağımlı değişkendeki log dönüştürülmüş tahminlemelere bakıyoruz

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_pred
# Bağımlı değişkendeki gözlemlerin tahminlemiş halleri geliyor (log dönüştürülmüş halleri geldi tabi)
# gerçek değerlerle karşılaştırma yapabilmek için bu log dönüşümünün tekrar tersini (inverse) almamız gerekmektedir.


# In[386]:


# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# In[387]:


# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_pred için)
new_y = np.expm1(y_pred)
new_y
# burada y_pred değerleri log dönüşümü yapılmış hedef değişken tahminlerini gösterirken
# new_y değeri y_pred in inverse uygulanmış yani log dönüşümünün tersinin yapılmış halinin tahmin sonuçlarını göstermektedir.
# bu iki değerlerin çıktılarını yani log dönüşümlü ve dönüşümsüz hallerini karşılaştırabilirsiniz


# In[388]:


# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_test için)
new_y_test = np.expm1(y_test)
new_y_test


# In[389]:


# Inverse alınan yani log dönüşümü yapılan tahminlerin RMSE değeri
np.sqrt(mean_squared_error(new_y_test, new_y))


# ### Hiperparametre optimizasyonlarını gerçekleştiriniz.

# In[390]:


lgbm_model = LGBMRegressor(random_state=46)


# In[391]:


rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse


# In[392]:


lgbm_model.get_params()


# In[393]:


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}


# In[394]:


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)


# In[395]:


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)


# In[396]:


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse


# ### Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

# In[397]:


# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:50])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


# In[398]:


#lightgbm modeli ile plot importance grafiğini çıkartıyoruz
model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)


# ## test dataframe indeki boş olan salePrice değişkenlerini tahminleyiniz

# In[429]:


print(train_data.drop(["index", "SalePrice"], axis=1).columns)
print(test_data.drop(["index", "SalePrice"], axis=1).columns)


# In[434]:


train_data = data[data['SalePrice'].notnull()]  # SalePrice dolu olan satırlar eğitim verisi
test_data = data[data['SalePrice'].isnull()]  # SalePrice boş olan satırlar test verisi


# In[436]:


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# In[437]:


model = LGBMRegressor()
model.fit(X, y)


predictions = model.predict(test_data.drop(["index","SalePrice"], axis=1))



# In[438]:


dictionary = {"index":test_data.index, "SalePrice":predictions}   # bir sözlük oluşturduk. Bu sözlük, test veri setinin indekslerini "Id" olarak ve tahmin edilen değerleri "SalePrice" olarak içeriyor. Bu, her tahminin hangi evle ilişkili olduğunu belirlemek için kullanılır.
dfSubmission = pd.DataFrame(dictionary)  #  sözlüğü bir pandas DataFrame'ine dönüştürüyoruz
dfSubmission.to_csv("housePricePredictions.csv", index=False)  #  tahmin sonuçlarını "housePricePredictions.csv" adlı bir CSV dosyasına kaydediyoruz. index=False parametresi, DataFrame'in indekslerinin dosyaya kaydedilmemesini sağlar.


# In[ ]:




