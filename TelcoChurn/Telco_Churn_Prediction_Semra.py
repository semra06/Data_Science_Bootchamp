#!/usr/bin/env python
# coding: utf-8

# In[546]:


import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.simplefilter(action="ignore")


def load_data_churn():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    return data

df = load_data_churn()
df.head()


# In[547]:


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

check_df(df)


# In[548]:


df.shape


# In[549]:


df.describe().T


# In[550]:


df['SeniorCitizen'].isnull().sum()


# In[551]:


## Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Boş veya geçersiz verileri kontrol edelim
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Hatalı verileri (NaN) kontrol edelim
print(df['TotalCharges'].isnull().sum())


# In[552]:


## Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
## Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
## Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
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
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(num_cols)


# In[553]:


## Numerik ve kategorik değişkenleri yakala
numerik_deg = df.select_dtypes(include=['int64', 'float64']).columns
kategorik_deg = df.select_dtypes(include=['object']).columns

print("Numeric Veriable:", numerik_deg)
print("Categoric Veriable:", kategorik_deg)


# In[554]:


df["Churn"].value_counts()


# In[555]:


df.shape


# In[556]:


## Öncelikle numerik ve kategorik değişkenlerin analizini yapılır.
## 1. olarak Kategorik değişkenlerin analizi yapılır 
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df, col)


# In[557]:


## 2. olarak değişkenlerin analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
for col in num_cols:
    num_summary(df, col, plot=True)


# In[558]:


## En son Numerik değişkenlerin target'a (churn) göre analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "std"]}))
    
for col in num_cols:
    target_summary_with_num(df, target="Churn", numerical_col=col)
 


# In[559]:


# 'Yes' ve 'No' değerlerini sayısal değerlere dönüştürelim
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Dönüşümü kontrol edelim
print(df['Churn'].head(10))
df.head()


# In[560]:


## Verimizi temizlemeden önce bir modelleme bakarız ki, veri temizleme işlemleri yapıldıktan sonra, 
## modellememiz eskiye göre ne kadar başarılı görmek için. 
## BASE MODEL oluşturunuz.
# Sayısal sütunları ve 'Churn' sütununu birlikte seçelim
df = df[num_cols + ['Churn']]

# Seçimi kontrol edelim
print(df.head())


# In[561]:


## Verimizi temizlemeden önce bir modelleme bakarız ki, veri temizleme işlemleri yapıldıktan sonra, 
## modellememiz eskiye göre ne kadar başarılı görmek için. 
## BASE MODEL oluşturunuz
y = df["Churn"]
X = df.drop("Churn", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
## Outcome değerlerine bak, eşitse Accuracy değeri en önemli kriter oluyor. 
## Veriler 0 ve 1 ler eşit değilse, diğer metriklere bakılır.


# In[597]:


## Korelasyon analizi yapacağız.
##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte


# Churn ile en yüksek korelasyona sahip değişkenlere bakalım:

df.corrwith(df["Churn"]).sort_values(ascending=False)

# Aylık ödeme ve müşteri yaşı arttıkça churn'ün arttığını görüyoruz!


# ## Feature Engineering

# In[563]:


## EKSİK DEĞER ANALİZİ
## Eksik değerleri bul
df = load_data_churn()
df.head()


# In[564]:


def missing_values_table(dataframe, na_name=False):
    ## eksik değer isimleri
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    ## eksik değer sayısı
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ## eksik değer oranı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    ## missing_df adında dataframe çevir, concat et yani birleştir. Değişkenin isimleri, sütunlara göre birleştir.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Hatalı verileri (NaN) kontrol edelim
    print(df['TotalCharges'].isnull().sum())
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


# In[565]:


# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Churn", na_columns)


# In[566]:


# 'TotalCharges' sütunundaki eksik değerleri sil
df = df.dropna(subset=['TotalCharges'])
df = df.drop('customerID', axis=1)


# In[598]:


## DAHA SONRA AYKIRI DEĞER ANALİZİ YAPIYORUZ
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# Sadece sayısal sütunlar üzerinde aykırı değer analizi yapalım
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    print(f"{col}: {outlier_thresholds(df, col)}")


# In[599]:


## Aykırı Değer Analizi
## aykırı değer var mı yok mu sorusunu soran fonksiyon
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
# Sadece 2'den fazla eşsiz değeri olan sayısal sütunlar için aykırı değer kontrolü
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].nunique() > 2:  # Sadece 0 ve 1 dışındaki değerler için
        if check_outlier(df, col):
            print(f"{col} sütununda aykırı değerler mevcut.")
    else:
        print(" 0 ve 1 değeri dışındaki numeric datalarda aykırı değerler mevcut değil.")


# In[602]:


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


# In[603]:


## Yeni özellik ekleme
# tenure süresine göre gruplandırma
def tenure_group(tenure):
    if tenure <= 12:
        return 'Yeni Müşteri'
    elif tenure <= 24:
        return 'Orta Süreli Müşteri'
    else:
        return 'Uzun Süreli Müşteri'

# Yeni bir 'tenure_group' değişkeni ekleyelim
df['tenure_group'] = df['tenure'].apply(tenure_group)

# Aylık ortalama ödeme hesaplama,avg_monthly_charge
df['avg_monthly_charge'] = df['TotalCharges'] / df['tenure']

# Yaşlı ve kadın olan müşterileri işaretleyelim, is_senior_female
df['is_senior_female'] = ((df['gender'] == 'Female') & (df['SeniorCitizen'] == 1)).astype(int)

df.head()


# In[604]:


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()


# In[605]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
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
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
print(num_cols)


# In[606]:


# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    #dataframe = dataframe.astype(int)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()


# In[607]:


##################################
# STANDARTLAŞTIRMA
##################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


# In[608]:


df["Churn"].value_counts()


# In[609]:


## Veri temizlemeden sonra Model oluşturunuz.
y = df["Churn"]
X = df.drop("Churn", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


# In[610]:


#Accuracy: 0.76
#Recall: 0.577
#Precision: 0.46
#F1: 0.51
#Auc: 0.7

#Accuracy: 0.78
#Recall: 0.602
#Precision: 0.5
#F1: 0.54
#Auc: 0.72


# ## Görev 3 : Modelleme 

# ### LogisticRegression

# In[611]:


from sklearn.linear_model import LogisticRegression

y = df["Churn"]

X = df.drop(["Churn"], axis=1)

## LogisticRegression-> Model kurma fonksiyonu bağımlı ve bağımsız değişkenleri ver. 
log_model = LogisticRegression().fit(X, y)
## bias, b sabitimizi buluyoruz. 
log_model.intercept_
## diğer değişkenlerin ağırlıklarını kat sayısını bulmak istersek.
log_model.coef_
## Burada diyabet hastası mı değilmi tahminde bulunduruyoruz.
y_pred = log_model.predict(X)
## y gerçek değerleri ifade eder. y_pred tahmin ifade eder.
y_pred[0:10]
y[0:10]


# In[612]:


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))


# In[613]:


#Accuracy: 0.76
#Recall: 0.577
#Precision: 0.46
#F1: 0.51
#Auc: 0.7

#Accuracy: 0.78
#Recall: 0.602
#Precision: 0.5
#F1: 0.54
#Auc: 0.72


# In[614]:


y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


# In[615]:


# Model Validation: 10-Fold Cross Validation
y = df["Churn"]
X = df.drop(["Churn"], axis=1)

log_model = LogisticRegression().fit(X, y)
## cross validate 5 defa train test yap
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


# In[616]:


print("Test_accuracy",cv_results['test_accuracy'].mean())
print("test_precision", cv_results['test_precision'].mean())
print("test_recall", cv_results['test_recall'].mean())
print("test_f1", cv_results['test_f1'].mean())
print("test_roc_auc", cv_results['test_roc_auc'].mean())


# In[617]:


######################################################
# Prediction for A New Observation
######################################################
## Rastgele bir veri geldiğinde, churn mü değil mi ona bakıyoruz. 
X.columns
random_user = X.sample(1)
print(random_user)
log_model.predict(random_user)


# ### KNN

# In[618]:


import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# In[619]:


y = df["Churn"]
X = df.drop(["Churn"], axis=1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)


# In[620]:


################################################
# 3. Modeling & Prediction
################################################
## KNN modeli kuruyoruz. Komşuluk sayısı hiperpaametre var. komşuluk sayısını bilmiyoruz. Model'i bu şekilde kuruyoruz. 
knn_model = KNeighborsClassifier().fit(X, y)
## veri setinden rast gele örneklem seçiyoruz.
random_user = X.sample(1, random_state=45)
## diabet hastası mı değil mi tahmin ediyoruz. 
knn_model.predict(random_user)


# In[621]:


################################################
# 4. Model Evaluation
################################################
y_pred = knn_model.predict(X)
y_prob = knn_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))


# In[622]:


#sınıflandırma modellerinin performansını ölçmek için kullanılan önemli bir metriktir
roc_auc_score(y, y_prob)


# In[623]:


# Logistic regration 0.8480177565044607


# In[624]:


print("Test_accuracy", cv_results['test_accuracy'].mean())
print("Test_f1", cv_results['test_f1'].mean())
print("Test_roc_auc", cv_results['test_roc_auc'].mean())


# In[625]:


# Test_accuracy 0.8036109838937804
# test_f1 0.598709114934933
# test_roc_auc 0.8451447600109627


# In[626]:


## ön tanımlı değerler
knn_model.get_params()


# In[627]:


################################################
# 5. Hyperparameter Optimization
################################################
## Hyperparameter kullanıcılar dışarıdan ayarlanması gereken parametreler vardır.
## GridSearchCV-> yöntemini kullanarak KNN algoritması için optimum komşuluk sayısının ne olduğunu bulacak bu yöntem
knn_model = KNeighborsClassifier()
## ön tanımlı değerler
knn_model.get_params()

## dışarıdan girilen parametreler, ifade ediliş tarzı sözlüktür.range(2, 50)-> bunları ara ne ile GridSearchCV ile.
## GridSearchCV method'u komşuluk için KNN modeli kurup hatamıza bakmamızı sağlar.
knn_params = {"n_neighbors": range(2, 50)}
## Hatamızı değerlendirmek için cross validation yaptık ayrı konu, hiper parametre  optimizasyonu için de  cross validation kullanıyoruz.
## ayrı bir konu. GridSearchCV hatamıza bakıyoruz, hatamıza bakarken de cross validation kullanarak 5 katlı şekilde hatamıza bak.
##  n_jobs=-1 yaparsak işlemciyi tam performans ile kullanır.Daha hızlı sonuçlara gitmemizi sağlar. verbose=1 rapor istediğimizi belirt.
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
## Best dememizin sebebi hatayı veren en düşük hatayı veren komşu sayısı gelecektir.
knn_gs_best.best_params_


# In[628]:


################################################
# 6. Final Model
################################################
## KNN değerinin en iyi değerini bulduk, şimdi bununla tekrar model kurmamız lazım. 2 yıldız demek atamak demektir.
## knn_final model kuruldu.
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
## final model'imizi kontrol et
cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

## rastgele kullanıcıyı kontrol et
random_user = X.sample(1)

knn_final.predict(random_user)


# In[642]:


##### EK MODELLEME YAPIYORUZ##
## En az 4 adet model oluşturma 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
y = df["Churn"]
X = df.drop(["Churn"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# In[ ]:





# In[ ]:




