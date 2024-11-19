##############################################################################
# PROJECT - BIKE SHARING
##############################################################################

# Gelecek Saatlerdeki Kiralanacak Bisiklet Sayısının Tahmini:
# https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset/data

# Proje Tanımı: Bisiklet paylaşım istasyonlarında belirli saatlerde kaç bisikletin kiralanacağını tahmin etmek,
# sistemi optimize etmek ve talebe göre bisiklet dağıtımını planlamak için kullanılabilir.
# Hava durumu, saat, gün, hafta sonu olup olmadığı gibi değişkenler bu modelde kullanılabilir.
# Tahmin: Kiralanan bisiklet sayısı.
# Model: Zaman serisi analizi veya regresyon
# Tahmin Değişkeni (Hedef Sütun): cnt (total count of rented bikes including both casual and registered)

# instant: Her gözleme verilen benzersiz bir indeks numarası.
# dteday: Gözlem tarihini gösteren tarih bilgisi.
# season: Mevsim bilgisi (1: İlkbahar, 2: Yaz, 3: Sonbahar, 4: Kış).
# yr: Yıl bilgisi (0: 2011, 1: 2012).
# mnth: Ay bilgisi (1: Ocak, 2: Şubat, vb.).
# hr: Saat bilgisi (0 - 23 arası saat dilimleri).
# holiday: Gözlemin tatil gününe denk gelip gelmediği (0: Hayır, 1: Evet).
# weekday: Haftanın günü (0: Pazar, 1: Pazartesi, vb.).
# workingday: Gözlemin iş gününe denk gelip gelmediği (0: Tatil veya hafta sonu, 1: İş günü).
# weathersit: Hava durumu
# (1: Açık, az bulutlu, 2: Bulutlu, sisli, 3: Hafif yağmurlu veya kar yağışlı, 4: Sağanak yağışlı, dolu, vb.).
# temp: Gerçek sıcaklık, normalize edilmiş (ölçeklendirilmiş) değer.
# atemp: Hissedilen sıcaklık, normalize edilmiş (ölçeklendirilmiş) değer.
# hum: Nem oranı, normalize edilmiş (ölçeklendirilmiş) değer.
# windspeed: Rüzgar hızı, normalize edilmiş (ölçeklendirilmiş) değer.
# casual: Kayıtsız kullanıcı sayısı (sisteme üye olmayan kullanıcıların kiraladığı bisiklet sayısı).
# registered: Kayıtlı kullanıcı sayısı (sisteme üye olan kullanıcıların kiraladığı bisiklet sayısı).
# cnt: Toplam kiralanan bisiklet sayısı (casual ve registered toplamı).

##############################################################################

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Modelleme ve Değerlendirme
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Model Seçimi ve Hata Metrikleri
from sklearn.metrics import (mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV

# Ölçekleme
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# İstatistiksel Analizler
from scipy import stats

# Uyarıların Kapatılması
warnings.filterwarnings("ignore")

from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 vasamak göster
pd.set_option('display.width', 700)  # konsolda gösterimi geniş tut
pd.set_option('display.expand_frame_repr', False)  # tüm sütunları tek bir satırda göster
pd.set_option('display.max_rows', 100)  # Satır limitini 100'e ayarlayın


df_ = pd.read_csv("datasets/hour.csv")
df = df_.copy()

df.head()
# instant      dteday  season  yr  mnth  hr  holiday  weekday  workingday  weathersit  temp  atemp   hum  windspeed  casual  registered  cnt
# 0        1  2011-01-01       1   0     1   0        0        6           0           1 0.240  0.288 0.810      0.000       3          13   16
# 1        2  2011-01-01       1   0     1   1        0        6           0           1 0.220  0.273 0.800      0.000       8          32   40
# 2        3  2011-01-01       1   0     1   2        0        6           0           1 0.220  0.273 0.800      0.000       5          27   32
# 3        4  2011-01-01       1   0     1   3        0        6           0           1 0.240  0.288 0.750      0.000       3          10   13
# 4        5  2011-01-01       1   0     1   4        0        6           0           1 0.240  0.288 0.750      0.000       0           1    1


df.shape  # (17379, 17)


###############################################################################
# Keşifçi Veri Analizi
###############################################################################

def check_df(dataframe, head=5):
    print("##################### Shape #######################")
    print(dataframe.shape)
    print("\n##################### Types #####################")
    print(dataframe.dtypes)
    print("\n##################### Nunique ###################")
    print(dataframe.nunique())
    print("\n##################### Head ######################")
    print(dataframe.head())
    print("\n##################### Tail ######################")
    print(dataframe.tail())
    print("\n##################### NaN #######################")
    print(dataframe.isnull().sum())
    print("\n################### Describe ####################")
    print(dataframe.describe().T)
    print("\n################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df.dtypes

df['dteday'] = pd.to_datetime(df["dteday"])

df.head()

def grab_col_names(dataframe, cat_th=25, car_th=20):  # ay ve saati de cat alsın diye
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal (bilgi taşımayan) değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir. (ör: Survived, Pclass)

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal (çok fazla sınıfı olan) değişkenler için sınıf eşik değeri (ör: Name, Ticket)

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  # tipi kategorik olanların hepsi

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  # tipi numerik görünen ama aslında kategorik olanlar (ör: Survived vs)

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  # tipi kategorik görünen ama kardinal olanlar (ör: name)

    cat_cols = cat_cols + num_but_cat  # tüm kategorikler + kardinalleri de içerir.

    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # kardinalleri de çıkarttık sadece kategorik kaldı

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # tipi numerik olanların hepsi

    num_cols = [col for col in num_cols if col not in num_but_cat]  # tüm num. olanlardan; num. görünüp cat.'leri çıkart

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car  # ay ve #


cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)
# ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
print(num_cols)
# ['instant', 'dteday', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
print(cat_but_car)
# []

num_cols = [col for col in num_cols if col not in ["dteday", "cnt"]]
# ['instant', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']


#########################
# Numerik ve kategorik değişkenlerin veri içindeki dağılımları:
#########################

# Kategorik değişkenlerin sınıflarını ve bu sınıfların oranları:

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        plt.figure()  # Yeni bir figür oluştur
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(f"Distribution of {col_name}")  # Başlık ekle
        plt.show(block=True)  # Zorla göster


for col in cat_cols:
    cat_summary(df, col, plot=True)


# ##########################################
#         season  Ratio
# season
# 3         4496 25.870
# 2         4409 25.370
# 1         4242 24.409
# 4         4232 24.351
# ##########################################
#       yr  Ratio
# yr
# 1   8734 50.256
# 0   8645 49.744
# ##########################################
#       mnth  Ratio
# mnth
# 5     1488  8.562
# 7     1488  8.562
# 12    1483  8.533
# 8     1475  8.487
# 3     1473  8.476
# 10    1451  8.349
# 6     1440  8.286
# 4     1437  8.269
# 9     1437  8.269
# 11    1437  8.269
# 1     1429  8.223
# 2     1341  7.716
# ##########################################
#      hr  Ratio
# hr
# 17  730  4.200
# 16  730  4.200
# 13  729  4.195
# 15  729  4.195
# 14  729  4.195
# 12  728  4.189
# 22  728  4.189
# 21  728  4.189
# 20  728  4.189
# 19  728  4.189
# 18  728  4.189
# 23  728  4.189
# 11  727  4.183
# 10  727  4.183
# 9   727  4.183
# 8   727  4.183
# 7   727  4.183
# 0   726  4.177
# 6   725  4.172
# 1   724  4.166
# 5   717  4.126
# 2   715  4.114
# 4   697  4.011
# 3   697  4.011
# ##########################################
#          holiday  Ratio
# holiday
# 0          16879 97.123
# 1            500  2.877
# ##########################################
#          weekday  Ratio
# weekday
# 6           2512 14.454
# 0           2502 14.397
# 5           2487 14.310
# 1           2479 14.264
# 3           2475 14.241
# 4           2471 14.218
# 2           2453 14.115
# ##########################################
#             workingday  Ratio
# workingday
# 1                11865 68.272
# 0                 5514 31.728
# ##########################################
#             weathersit  Ratio
# weathersit
# 1                11413 65.671
# 2                 4544 26.146
# 3                 1419  8.165
# 4                    3  0.017
# ##########################################


# Numerik değişkenlerin yüzdelik değerleri
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        if dataframe[col_name].dtype == "object" or dataframe[col_name].nunique() < 20:  # Kategorik değişken kontrolü
            plt.figure()  # Yeni bir figür oluştur
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.title(f"Distribution of {col_name}")  # Başlık ekle
            plt.show()
        else:  # Sayısal değişkenler için histogram
            plt.figure()
            sns.histplot(dataframe[col_name], bins=20, kde=True)
            plt.title(f"Histogram of {col_name}")  # Başlık ekle
            plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)

# casual ve registered hariç diğer değişkenler standartlastırılmıs veye normalize edildiği için uygun görünüyor.
# ama casual ve registered değişkenlerinde aykırı değer var gibi duruyor.


#########################
# Numerik değişkenler ile hedef değişkenin incelemesi:
#########################

# Hedef değişken numerik olduğu için bu inceleme gereksiz oluyor.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n")
    print("##########################################")


for col in num_cols:
    target_summary_with_num(df, "cnt", col)


#########################
# Kategorik değişkenler ile hedef değişkenin incelemesi:
#########################

# Kategorik değişkenler ile hedef değişken incelemesi Fonksiyonu: !
def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET_FREQUENCY": dataframe.groupby(cat_col)[target].count(),
                        "RATIO": 100 * dataframe.groupby(cat_col)[target].count() / len(dataframe),
                        "TARGET_MEAN": dataframe.groupby(cat_col)[target].mean()}))
    print("##################################################")


for col in cat_cols:
    target_summary_with_cat(df, "cnt", col)
#         TARGET_FREQUENCY  RATIO  TARGET_MEAN
# season
# 1                   4242 24.409      111.115
# 2                   4409 25.370      208.344
# 3                   4496 25.870      236.016
# 4                   4232 24.351      198.869
# ##################################################
#     TARGET_FREQUENCY  RATIO  TARGET_MEAN
# yr
# 0               8645 49.744      143.794
# 1               8734 50.256      234.666
# ##################################################
#       TARGET_FREQUENCY  RATIO  TARGET_MEAN
# mnth
# 1                 1429  8.223       94.425
# 2                 1341  7.716      112.865
# 3                 1473  8.476      155.411
# 4                 1437  8.269      187.261
# 5                 1488  8.562      222.907
# 6                 1440  8.286      240.515
# 7                 1488  8.562      231.820
# 8                 1475  8.487      238.098
# 9                 1437  8.269      240.773
# 10                1451  8.349      222.159
# 11                1437  8.269      177.335
# 12                1483  8.533      142.303
# ##################################################
#     TARGET_FREQUENCY  RATIO  TARGET_MEAN
# hr
# 0                726  4.177       53.898
# 1                724  4.166       33.376
# 2                715  4.114       22.870
# 3                697  4.011       11.727
# 4                697  4.011        6.353
# 5                717  4.126       19.890
# 6                725  4.172       76.044
# 7                727  4.183      212.065
# 8                727  4.183      359.011
# 9                727  4.183      219.309
# 10               727  4.183      173.669
# 11               727  4.183      208.143
# 12               728  4.189      253.316
# 13               729  4.195      253.661
# 14               729  4.195      240.949
# 15               729  4.195      251.233
# 16               730  4.200      311.984
# 17               730  4.200      461.452
# 18               728  4.189      425.511
# 19               728  4.189      311.523
# 20               728  4.189      226.030
# 21               728  4.189      172.315
# 22               728  4.189      131.335
# 23               728  4.189       87.831
# ##################################################
#          TARGET_FREQUENCY  RATIO  TARGET_MEAN
# holiday
# 0                   16879 97.123      190.429
# 1                     500  2.877      156.870
# ##################################################
#          TARGET_FREQUENCY  RATIO  TARGET_MEAN
# weekday
# 0                    2502 14.397      177.469
# 1                    2479 14.264      183.745
# 2                    2453 14.115      191.239
# 3                    2475 14.241      191.131
# 4                    2471 14.218      196.437
# 5                    2487 14.310      196.136
# 6                    2512 14.454      190.210
# ##################################################
#             TARGET_FREQUENCY  RATIO  TARGET_MEAN
# workingday
# 0                       5514 31.728      181.405
# 1                      11865 68.272      193.208
# ##################################################
#             TARGET_FREQUENCY  RATIO  TARGET_MEAN
# weathersit
# 1                      11413 65.671      204.869
# 2                       4544 26.146      175.165
# 3                       1419  8.165      111.579
# 4                          3  0.017       74.333
# ##################################################


#########################
# Kategorik değişkenler ile hedef değişkenin Grafiksel olarak incelemesi:
#########################

### "Mevsimlere Göre Bisiklet Kiralama Oranı "
def plot_custom_pie_chart_with_labels(dataframe, target, categorical_col):
    target_mean = dataframe.groupby(categorical_col)[target].mean()

    # Kategorilere ait açıklamalar
    category_labels = { 1:'Kış', 2: 'İlkbahar', 3: 'Yaz', 4: 'Sonbahar'}

    # Yeni etiketleri oluşturmak
    labels = [f'{season}: {category_labels.get(season)}' for season in target_mean.index]

    # Pasta grafiği oluşturulması
    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(
        target_mean.values,
        labels=labels,  # Yeni etiketler
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Paired.colors,  # Renk paleti
        wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'},  # Dilim kenarlıkları
        explode=(0, 0, 0.1, 0)  # en çok tercihe edilen  dilimi ayırma
    )

    # Etiketleri daha belirgin yapmak için
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('black')

    # Başlık
    plt.title(f'Target Mean Distribution by {categorical_col}', fontsize=14, fontweight='bold')
    plt.title("Mevsimlere Göre Bisiklet Kiralama Oranı ",fontsize=14, fontweight='bold')

    # Grafik görünümünü eşit yapmak
    plt.axis('equal')

    # Grafiği göster
    plt.show()


plot_custom_pie_chart_with_labels(df, "cnt", "season")


### "Yıllara Göre Kiralanan Bisiklet Sayısı"
def plot_target_mean_bar_chart_with_black_values(dataframe, target, categorical_col):
    # Ortalamayı hesapla
    target_mean = dataframe.groupby(categorical_col)[target].mean()

    # Yıl etiketlerini değiştirme
    year_labels = {0: '2011', 1: '2012'}
    target_mean.index = target_mean.index.map(year_labels)

    # Grafik oluşturulması
    plt.figure(figsize=(5, 5))
    ax = sns.barplot(x=target_mean.index, y=target_mean.values, palette='viridis')

    # Başlık ve etiketler
    #plt.title(f'Target Mean by {categorical_col}', fontsize=14, fontweight='bold')
    plt.title("Yıllara Göre Kiralanan Bisiklet Sayısı",fontsize=14, fontweight='bold')
    plt.xlabel('Yıl', fontsize=12)
    plt.ylabel('Kiralanan Bisiklet Sayısı', fontsize=12)

    # Her bir çubuğun üzerine değeri ekleme (siyah renkte ve en üstte)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='top',  # 'bottom' değeri çubuğun üst kısmına yerleştirir
                    fontsize=12,fontweight='bold', color='black',
                    xytext=(0, 8), textcoords='offset points')

    # Grafiği göster
    plt.show()


plot_target_mean_bar_chart_with_black_values(df, 'cnt', 'yr')


### "Aylara Göre Kiralanan Bisiklet Sayısı
def plot_target_mean_monthly(dataframe, target, categorical_col):
    # Ortalamayı hesapla
    target_mean = dataframe.groupby(categorical_col)[target].mean()

    # Ay isimlerini değiştirme
    month_labels = {
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
        7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    }
    target_mean.index = target_mean.index.map(month_labels)

    # Grafik oluşturulması
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=target_mean.index, y=target_mean.values, palette='coolwarm')

    # Başlık ve etiketler
    plt.title("Aylara Göre Kiralanan Bisiklet Sayısı",fontsize=14, fontweight='bold')
    #plt.title(f'Target Mean by Month', fontsize=14, fontweight='bold')
    plt.xlabel('Aylar', fontsize=12)
    plt.ylabel('Kiralanan Bisiklet Sayısı', fontsize=12)

    # Her bir çubuğun üzerine değeri ekleme (siyah renkte ve en üstte)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='top',  # 'bottom' değeri çubuğun üst kısmına yerleştirir
                    fontsize=12, color='black',
                    xytext=(0, 8), textcoords='offset points')

    # Grafiği göster
    plt.show()


plot_target_mean_monthly(df, 'cnt', 'mnth')


### "Tatil ve Tatil Olmayan Günlere Göre Bisiklet Kiralama Sayısı"
# Veriler
holiday_labels = ['0: No Holiday', '1: Holiday']
holiday_means = [190.429, 156.870]

# Grafik boyutunu ayarlama
plt.figure(figsize=(6, 2))

# Yatay bar grafiği çizdirme
plt.barh(holiday_labels, holiday_means, color=['steelblue', 'coral'])

# Başlık ve eksen etiketleri
plt.title("Tatil ve Tatil Olmayan Günlere Göre Bisiklet Kiralama Sayısı")
plt.xlabel("Bisiklet Kiralama Sayısı")

# Değerleri çubukların yanına ekleme
for index, value in enumerate(holiday_means):
    plt.text(value + 3, index, f"{value:.2f}", va='center', color='black')

plt.show()


### Nem Bilgisi ile Bisiklet Kiralama Arasındaki İlişki

sns.regplot(x=df['hum'], y=df['cnt'])
plt.title('relation between humidity and count')
plt.show()
sns.regplot(x=df['temp'], y=df['cnt'])
plt.title('relation between humidity and count')
plt.show()


#########################
# Korelasyon Analizi:
#########################

df_corr = df.corr()

# Heatmap'i çiz
plt.figure(figsize=(12, 10))  # Görüntü boyutunu ayarlayabilirsiniz
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Korelasyon Matrisi Heatmap")
plt.show()


# Yüksek Korelasyona sahip değişkenlerin analizi:
def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool kullanıldı
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df, plot=False)
# ['dteday', 'yr', 'mnth', 'atemp', 'cnt']


#########################
# Çarpıklık (SKEWNESS) Analizi:
#########################

from scipy import stats

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return


plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(df, 'cnt')
plt.subplot(6, 1, 2)
check_skew(df, 'temp')
plt.subplot(6, 1, 3)
check_skew(df, 'hum')
plt.subplot(6, 1, 4)
check_skew(df, 'windspeed')
plt.subplot(6, 1, 5)
check_skew(df, 'casual')
plt.subplot(6, 1, 6)
check_skew(df, 'registered')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)

# Görselleri kaydet:
# plt.savefig('before_transform.png', format='png', dpi=1000)

# cnc hedef değişkeninde sağa çarpıklık durumu söz konusu. Log dönüşümü yapılacak.

df["cnt"].describe().T


#########################
# Aykırı gözlem durumu:
#########################

# Veri setinde değişkenlerin eşik değerlerinin hesaplanması:
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Veri setinde değişkenlerde aykırı değer var mı yok mu?:
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))
# instant False
# temp False
# atemp False
# hum False
# windspeed False
# casual True --> aykırı değer bulunuyor ama hedef değişkenini etkileyeceği için baskılama yapılmayacak
# registered False

# cnt = casual + registered

# Değişkenin aykırı değer sayısının hesaplanması:
def count_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    return outliers.shape[0]  # This will return the number of outliers


casual_outliers = count_outliers(df, "casual")
print(f"Number of casual outliers: {casual_outliers}")  # Number of casual outliers: 10


## GRAFİKTE GÖZLEMLEYELİM
def plot_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    plt.figure(figsize=(12, 6))

    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(dataframe[col_name])
    plt.axhline(y=up_limit, color='r', linestyle='--')
    plt.axhline(y=low_limit, color='r', linestyle='--')
    plt.title(f'Boxplot of {col_name}')

    # Distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(dataframe[col_name], kde=True)
    plt.axvline(x=up_limit, color='r', linestyle='--')
    plt.axvline(x=low_limit, color='r', linestyle='--')
    plt.title(f'Distribution of {col_name}')

    plt.tight_layout()
    plt.show(block=True)


plot_outliers(df, "casual")


"""" Aykırı değere sahip değişkendeki aykırı değerlerin eşik değerleri ile değiştirilmesi:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit"""


#########################
# Eksik gözlem durumu:
#########################

df.isnull().sum()

# Veri setinde eksik gözlem bulunmuyor.


###############################################################################
# Feature Engineering
###############################################################################

# Saatlere göre aralık verme:
def get_time_of_day(hour):
    if 5 <= hour <= 9:
        return 'morning'  # Sabah
    elif 10 <= hour <= 15:
        return 'afternoon'  # Öğle
    elif 16 <= hour <= 19:
        return 'evening'  # Akşam
    else:
        return 'night'  # Gece


# Rush hour saatlerini ekleme:
def get_rush_hour(hour):
    if 7 <= hour <= 9:
        return 'morning_rush'  # Sabah Yoğun Saat
    elif 16 <= hour <= 19:
        return 'evening_rush'  # Akşam Yoğun Saat
    else:
        return 'no_rush'  # Yoğun Saat Değil


df['NEW_time_of_day'] = df['hr'].apply(get_time_of_day)
df['NEW_rush_hour'] = df['hr'].apply(get_rush_hour)

########################
# Hava Durumu Kategorileri
def weather_impact(weather):
    if weather == 1:                     ###açık
        return 'high'
    elif weather == 2:
        return 'medium'
    elif weather == 3:
        return 'low'
    else:
        return 'very_low'


df['NEW_weather_impact'] = df['weathersit'].apply(weather_impact)

########################
# Sıcaklık Kategorileri
def categorize_temp(temp):
    if temp <= 0:
        return 'Cold'
    elif 0 < temp <= 20:
        return 'Mild'
    elif 20 < temp <= 30:
        return 'Warm'
    else:
        return 'Hot'


df['NEW_temp_category'] = df['temp'].apply(categorize_temp)

########################
# Nem Kategorileri

def categorize_hum(hum):
    if hum <= 0.3:
        return 'Low Humidity'
    elif 0.3 < hum <= 0.6:
        return 'Medium Humidity'
    else:
        return 'High Humidity'


df['NEW_hum_category'] = df['hum'].apply(categorize_hum)

########################
# Rüzgar Hızı Kategorileri
def categorize_windspeed(windspeed):
    if windspeed <= 5:
        return 'Light Wind'
    elif 5 < windspeed <= 15:
        return 'Moderate Wind'
    else:
        return 'Strong Wind'


df['NEW_windspeed_category'] = df['windspeed'].apply(categorize_windspeed)

########################
# Casual kullanıcı sayısını 4 kategoriye ayırma
df['NEW_non_registered_user_category'] = pd.qcut(df['casual'], q=4,
                                                 labels=["Very Low Non-Registered", "Low Non-Registered",
                                                         "Moderate Non-Registered", "High Non-Registered"])

# Kayıtlı kullanıcı sayısını 4 kategoriye ayırma
df['NEW_registered_user_category'] = pd.qcut(df['registered'], q=4,
                                             labels=["Very Low Registered", "Low Registered",
                                                     "Moderate Registered", "High Registered"])

df.head(10)

df.drop(["instant", "dteday", "atemp", "casual", "registered"], inplace=True, axis=1)
# gereksiz değişkenler veri setinden çıkartıldı.

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################
# SineCosine Transformation:
#########################

df["weekday"] = df["weekday"] + 1
sincos = ["hr", "mnth", "weekday"]
df[sincos].head(10)

def sin_cos_encoding(dataframe, columns):
    for col in columns:
        max_val = dataframe[col].max()
        dataframe[f'{col}_sin'] = np.sin(2 * np.pi * dataframe[col] / max_val)
        dataframe[f'{col}_cos'] = np.cos(2 * np.pi * dataframe[col] / max_val)
    return dataframe


df = sin_cos_encoding(df, sincos)

# Sonuçların kontrolü
sinecosinelistesi = ["hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"]
df[sinecosinelistesi].head(10)

df.describe().T

df.drop(["hr", "mnth", "weekday"], inplace=True, axis=1)  # sin-cos transformasyonu uyguladığımız değişkenleri çıkarttık

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################
# MinMaxScaler:
#########################

# ÇOK ÖNEMLİ NOT:
# Sine Cosine transformasyonu sonrası türetilen değişkenlerin de mixmaxscaler'a dahil edilmesi gerekiyor

num_cols  # ['temp', 'hum', 'windspeed', 'cnt']
scale_num_cols = [col for col in num_cols if col not in "cnt"]
scale_num_cols.extend(sinecosinelistesi)

scaler = MinMaxScaler()  # Min-Max Scaler nesnesi oluşturuldu
df[scale_num_cols] = scaler.fit_transform(df[scale_num_cols])  # Yalnızca scale_num_cols ölçeklendirildi

df[scale_num_cols].head(10)  # Sonuçların kontrolü
df.head()


#########################
# One Hot Encoding:
#########################

cat_cols = [col for col in cat_cols if col not in sinecosinelistesi]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

# True/False değerlerini 1/0'a dönüştür
df = df.replace({True: 1, False: 0})

df.head(10)
df.shape  # (17379, 35)


#########################
# İlgili DataFramei df_model isimli yeni bir dataframe'e kopyalamak:
#########################

df_model = df.copy()
df_model.head()


#########################
# # LOG TRANSFORMATION:
#########################

df_model["cnt"].head(55)
df_model['cnt_log'] = np.log1p(df_model['cnt'])
df_model.drop("cnt", inplace=True, axis=1)
df_model.head()

# Hedef değişkenin histogram grafiği:
plt.hist(df_model["cnt_log"], bins=50)
plt.title('Target Variable (cnt_log) Distribution')
plt.xlabel('cnt_log')
plt.ylabel('Frequency')
plt.show()


#########################
# VIF (Variance Inflation Factor):
#########################

"""VIF (Variance Inflation Factor) Nedir?
VIF, değişkenler arasındaki kollinearlik derecesini ölçer. 
VIF değeri yüksek olan bir değişken, diğer değişkenlerle yüksek oranda ilişkili demektir.
Yani VIF, bir bağımsız değişkenin diğer bağımsız değişkenlerle ne kadar ilişkili olduğunu değerlendirir. 
VIF değeri yüksekse, o değişken diğer değişkenlerle güçlü bir şekilde ilişkilidir, 
bu da model performansını olumsuz etkileyebilir.

******
VIF, çoklu doğrusal regresyon modellerinde, bağımsız değişkenler arasındaki doğrusal ilişkileri değerlendirmek ve 
multicollinearity problemlerini tespit etmek için kullanılan güçlü bir araçtır. Yüksek VIF değerleri genellikle 
bir özelliğin başka özelliklerle yüksek ilişkiye girdiğini ve modelin yorumlanabilirliğini zorlaştırabileceğini gösterir
*******

VIF Değerlerinin Yorumlanması
VIF değerleri şu şekilde yorumlanabilir:

VIF < 5: Çoklu doğrusal bağlantı sorunu yoktur.
5 ≤ VIF < 10: Çoklu doğrusal bağlantı kısmen yüksektir. Dikkat edilmelidir.
VIF ≥ 10: Çoklu doğrusal bağlantı yüksektir ve bu değişkeni modelden çıkarmayı veya dönüşüm uygulamayı düşünmelisiniz.

VIF Ne Zaman Hesaplanmalıdır?
VIF analizi, model öncesi ve model sonrası yapılabilir, ancak genellikle model öncesinde yapılması daha yaygındır.

1. Model Öncesi Analiz
Amaç: Bağımsız değişkenler arasında çoklu doğrusal bağlantıyı tespit etmek ve modeli optimize etmektir.
Ne Zaman Yapılır?: Özellikle regresyon modelleri oluşturulmadan önce, 
değişkenlerin bağımsız olduğundan emin olmak için VIF analizi yapılır. 
Yüksek VIF değerine sahip değişkenler modelden çıkarılarak overfitting ve yanıltıcı sonuçların önüne geçilebilir.
Örnek Durum: Özellikle lineer regresyon gibi doğrusal modeller çoklu doğrusal bağlantıya karşı hassastır. 
Bu nedenle, bağımsız değişkenlerin birbirleriyle ilişkisinin düşük olduğundan emin olmak için VIF hesaplanmalıdır.

2. Model Sonrası Analiz
Amaç: Modelin performansını değerlendirdikten sonra, 
çoklu doğrusal bağlantı olup olmadığını kontrol etmek ve gerekirse değişiklikler yapmaktır.
Ne Zaman Yapılır?: Model eğitildikten sonra hala düşük performans veya yüksek hata oranları 
(örneğin, yüksek RMSE veya düşük R²) gözlemleniyorsa, bu durumda VIF analizi yapılabilir.
Örnek Durum: Özellikle düşük performans sergileyen veya aşırı uyum gösteren modellerde, 
VIF kullanılarak hangi değişkenlerin modeli etkileyebileceği tespit edilir ve gerekirse model optimize edilir.

VIF Analizinden Sonra Neler Yapılabilir?
VIF değeri yüksek olan değişkenler tespit edildiyse, şu çözümler uygulanabilir:

Yüksek VIF değerine sahip değişkenleri çıkarın: Çoklu doğrusal bağlantıya neden olan değişkenler modelden çıkartılabilir
Dönüşümler uygulayın: Örn, değişkenler arasında logaritmik dönüşümler veya standartlaştırma yapark bağlantı azaltılabilr
PCA (Principal Component Analysis): Yüksek korelasyona sahip değişkenleri, 
ana bileşen analizi ile daha az sayıda bağımsız değişkene indirgemek etkili olabilir.
Feature Engineering: Değişkenleri yeniden tanımlayarak veya birleştirerek yeni değişkenler oluşturabilirsiniz.
"""


# VIF hesaplaması için gerekli DataFrame'i oluşturuyoruz
vif = df_model[scale_num_cols]

# VIF hesaplaması
vif_data = pd.DataFrame()
vif_data['Feature'] = vif.columns
vif_data['VIF'] = [variance_inflation_factor(vif.values, i) for i in range(vif.shape[1])]

# VIF < 5: Çoklu doğrusal bağlantı sorunu yoktur.
# 5 ≤ VIF < 10: Çoklu doğrusal bağlantı kısmen yüksektir. Dikkat edilmelidir.
# VIF ≥ 10: Çoklu doğrusal bağlantı yüksektir ve bu değişkeni modelden çıkarmayı veya dönüşüm uygulamayı düşünmelisiniz.

# Sonuçları yazdırıyoruz
print(vif_data)
#        Feature    VIF
# 0         temp  6.989
# 1          hum 14.198  --> yüksek çoklu doğrusal bağlantı (multicollinearitry sorunu)
# 2    windspeed  3.717
# 3       hr_sin  3.772
# 4       hr_cos  3.185
# 5     mnth_sin  2.959
# 6     mnth_cos  3.574
# 7  weekday_sin  2.860
# 8  weekday_cos  2.564

corr = df[scale_num_cols].corr()
corr

df_model.drop("hum", inplace=True, axis=1)  # multicollinearity sorunundan dolayı hum'u çıkartıyoruz


#########################
# RFE (Recursive Feature Elimination):
#########################

"""RFE (Recursive Feature Elimination) Nedir?
Recursive Feature Elimination (RFE), özellik seçimi (feature selection) için kullanılan bir tekniktir. 
Özellikle gereksiz veya düşük önem düzeyine sahip özellikleri belirleyerek modelin performansını artırmayı amaçlar. 
RFE, modelin başarısına en fazla katkıda bulunan özellikleri bulmak için tekrarlayan bir şekilde özellikleri eler.

RFE Nasıl Çalışır?
Model Eğitimi: Başlangıçta, tüm özelliklerle bir model eğitilir (genellikle regresyon veya sınıflandırma modeli).
Özellik Önem Sıralaması: Model, her bir özelliğin önem derecesini hesaplar.
Özellik Elemek: En düşük önem düzeyine sahip olan bir veya birden fazla özellik çıkarılır.
Tekrar: Kalan özelliklerle model yeniden eğitilir ve önem dereceleri tekrar hesaplanır.
Durdurma Kriteri: Bu işlem, önceden belirlenmiş bir sayıda özellik kalana kadar veya 
model performansı belirli bir seviyeye ulaşana kadar devam eder.
Sonuç olarak, RFE özelliklerin önem sırasını belirler ve modelin en iyi performans gösterdiği özellik setini sunar.

RFE Model Öncesi mi Yoksa Model Sonrası mı Hesaplanır?
RFE, model oluşturulmadan önce, modelin performansını artırmak amacıyla uygulanır. Yani, model öncesi bir adımdır.

Model öncesi adım olarak, modelin başarısız olmasına neden olabilecek gereksiz veya 
düşük bilgi içeriğine sahip özellikleri ortadan kaldırarak modelin daha iyi genelleme yapmasını sağlar.
Model sonrası yapılması anlamlı değildir, çünkü model zaten tüm özelliklerle eğitilmiş olacaktır. 
Dolayısıyla, modelin eğitilmesi sırasında düşük performansa neden olabilecek özellikler zaten dahil edilmiştir.
RFE'nin Önemi ve Sağladığı Faydalar
Özellik Seçimi ve Model Basitleştirme:

Gereksiz özellikleri çıkararak modeli daha basit hale getirir.
Daha az özellik kullanarak modeli eğitmek, eğitim süresini kısaltır ve hesaplama maliyetlerini azaltır.
Overfitting'i Azaltma:

Çok sayıda gereksiz veya gürültülü değişken, modelin overfitting yapmasına neden olabilir. 
RFE, yalnızca en önemli değişkenleri seçerek overfitting riskini azaltır.
Model Performansını Artırma:

Model, yalnızca en önemli değişkenleri kullanarak eğitildiğinde, genellikle daha iyi genelleme performansı gösterir.
Yorumlanabilirlik:

Daha az ancak daha anlamlı özellik kullanıldığında, modelin çıktıları daha kolay yorumlanabilir hale gelir."""

from sklearn.feature_selection import RFE

# Veriyi hazırlama (hedef değişken 'cnt_log')
b = df_model['cnt_log']
A = df_model.drop(["cnt_log"], axis=1)

# Veriyi eğitim ve test setlerine ayırma
X_trainA, X_test, y_trainA, y_test = train_test_split(A, b, test_size=0.2, random_state=17)

# Lineer regresyon modelini başlatma
modelA = LGBMRegressor()

# RFE ile özellik seçimi yapma
rfe = RFE(modelA, n_features_to_select=12)  # Burada 12 özellik seçilecektir
X_train_rfe = rfe.fit_transform(X_trainA, y_trainA)

# Seçilen özellikleri yazdırma
selected_features = A.columns[rfe.support_]
print("Seçilen Özellikler:", selected_features)

# Seçilen Özellikler: Index(['temp', 'windspeed', 'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos',
# 'weekday_cos', 'New_weather_impact_low', 'yr_1', 'workingday_1',
# 'NEW_non_registered_user_category_Low Non-Registered',
# 'NEW_registered_user_category_Low Registered'], dtype='object')


#######################################################
# MODEL BUILDING:
#######################################################

y = df_model['cnt_log']  # hedef değişken
X = df_model.drop(["cnt_log"], axis=1)  # bağımsız değişkenler

models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")

""" "hum" değişkeni çıkartılmamış durumda hatalar:
# RMSE: 0.4016 (LR)
# RMSE: 0.3923 (KNN)
# RMSE: 0.3766 (CART)
# RMSE: 0.278 (RF)
# RMSE: 0.2922 (SVR)
# RMSE: 0.3129 (GBM)
# RMSE: 0.2875 (XGBoost)
# RMSE: 0.2688 (LightGBM) """

# RMSE: 0.4016 (LR)
# RMSE: 0.3909 (KNN)
# RMSE: 0.3767 (CART)
# RMSE: 0.2778 (RF)
# RMSE: 0.2915 (SVR)
# RMSE: 0.3124 (GBM)
# RMSE: 0.2834 (XGBoost)
# RMSE: 0.2698 (LightGBM) --> en düşük hata

m = df_model['cnt_log'].mean()  # 4.5747
# rmse_LGM = 0.2698
# (rmse_LGM * 100) / m  # 5.897 (%)

""" Hata Oranının Yorumu:
Bu hesaplama, LightGBM modelinin tahmin hatasının, log dönüşümü yapılmış bağımlı değişkenin ortalamasının yaklaşık 
%6'sı kadar olduğunu gösterir.

Başka bir deyişle:
Ortalama cnt_log değeri 4.5747 iken, model tahminlerinde ortalama 0.2698 birimlik bir hata yapıyor.
Bu hata, bağımlı değişkenin ortalamasının sadece %6'sı kadar.
Bu durum, modelin genel olarak iyi bir performans sergilediğini ve hataların bağımlı değişkenin genel seviyesine kıyasla 
oldukça küçük olduğunu gösteriyor.

Özetle:
RMSE'nin bağımlı değişken ortalamasına oranının %6 gibi düşük bir değer çıkması, modelin tahminlerinin oldukça başarılı 
olduğunu gösterir. Eğer hata oranı %20, %30 gibi yüksek bir seviyede olsaydı, 
bu modelin sonuçlarının güvenilir olmadığını ve iyileştirilmesi gerektiğini işaret ederdi."""


#########################
# FEATURE IMPORTANCE:
#########################

# LIGHT GBM ICIN
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)  # GRAFİĞİ ÇİZDİR


#########################
# LIGHTGBM ICIN HIPERPARAMETRE OPTIMIZASYONU:
#########################

# Train verisi ile model kurup, model başarısını değerlendiriniz.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

lgbm_model = LGBMRegressor(random_state=17)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)   # gridsearch
# Fitting 3 folds for each of 24 candidates, totalling 72 fits

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

lgbm_gs_best.best_params_  # {'colsample_bytree': 1, 'learning_rate': 0.1, 'n_estimators': 500}

rmse_hp = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse_hp  # 0.274

# BEFORE HIPERPARAMETRE RMSE: 0.2698 (LightGBM)
# AFTER  HIPERPARAMETRE RMSE: 0.2740 --> hiperparametre optimizasyonu sonrası rmse yükselmiş


#########################
# Modelin görmediği test verisindeki gerçek değerler ile modelin tahmin değerlerinin karşılaştırması:
#########################

y_pred = final_model.predict(X_test)  # Modelin görmediği %20'lik X_test bağımsız değişkenleri ile tahminliyoruz
# array([5.90025172, 5.87067451, 5.16351524, ..., 5.30409587, 5.69966408,
#        4.69712809])

# Gerçek değerler ve tahmin edilen değerleri bir arada görmek için bir DataFrame oluşturuyoruz
comparison_df = pd.DataFrame({'Actual': y_test,      # Modele sokmadığımız %20'lik bağımlı değişkenin gerçek değerleri
                              'Predicted': y_pred})  # Modele girmeyen X_test bağımsız değişkenleri ile yapıln tahminler
#        Actual  Predicted
# 5812    6.019      5.900
# 5061    5.775      5.871
# 9955    5.124      5.164
# 923     2.708      2.978
# 10939   4.920      4.917
# ...       ...        ...

# Tahmin başarı yüzdesini hesaplayıp yeni bir DataFrame'e ekleyelim
comparison_df['Prediction Accuracy (%)'] = (1 - abs(comparison_df['Predicted'] - comparison_df['Actual']) /
                                            comparison_df['Actual']) * 100

print(comparison_df)
#        Actual  Predicted  Prediction Accuracy (%)
# 5812    6.019      5.900                   98.034
# 5061    5.775      5.871                   98.335
# 9955    5.124      5.164                   99.228
# ...       ...        ...                      ...

print(f"First 20 rows of the comparison: \n{comparison_df.head(10)}")  # İlk 10 satırı göster
# First 20 rows of the comparison:
#        Actual  Predicted  Prediction Accuracy (%)
# 5812    6.019      5.900                   98.034
# 5061    5.775      5.871                   98.335
# 9955    5.124      5.164                   99.228
# 923     2.708      2.978                   90.046
# 10939   4.920      4.917                   99.936
# 16948   4.796      4.507                   93.984
# 9889    1.946      1.659                   85.262
# 17277   5.288      5.187                   98.084
# 5703    1.946      1.993                   97.604
# 14511   5.338      5.514                   96.686


#########################
# OVERFIT KONTROLÜ:
#########################

# Eğitim seti için tahmin yap ve RMSE'yi hesapla
y_train_pred = final_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 0.1703

# Test seti için tahmin yap ve RMSE'yi hesapla
y_test_pred = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 0.1684

# Sonuçları yazdır
print(f"Eğitim RMSE: {train_rmse:.4f}")  # Eğitim RMSE: 0.1703
print(f"Test RMSE: {test_rmse:.4f}")  # Test RMSE: 0.1684

# Overfitting kontrolü:
if test_rmse > train_rmse:
    print("Modelin test RMSE'si eğitim RMSE'sinden daha yüksek. Bu, overfitting olabileceğini gösterir.")
else:
    print("Model overfitting yapmıyor gibi görünüyor.")

# Model overfitting yapmıyor gibi görünüyor. (Test RMSE, Eğitim RMSE'den daha düşük --> 0.1684 < 0.1703)


#########################
# LOG dönüşümü yapılmış hedef değişken değerlerini orjinal değerlere çevirme işlemi:
#########################

# comparison_df, log'lu değerleri ve satır bazında tahmin yüzdesini içeren bir dataframedir
# log'lu değerlerin orjinallerini de (logdan kurtulmuş gerçek değerler) görüntüleyeceğimiz yeni bir DataFrame atayalım
df_log = pd.DataFrame(comparison_df)

# Log dönüşümünden orijinal değerlere dönme
df_log['Actual_original'] = np.expm1(df_log['Actual'])
df_log['Predicted_original'] = np.expm1(df_log['Predicted'])

print(df_log)
#        Actual  Predicted  Prediction Accuracy (%)  Actual_original  Predicted_original
# 5812    6.019      5.900                   98.034          410.000             364.129
# 5061    5.775      5.871                   98.335          321.000             353.488
# ...       ...        ...                      ...              ...                 ...

# Sütunların sırasını değiştirelim
df_log = df_log[["Actual", "Predicted", "Actual_original", "Predicted_original", "Prediction Accuracy (%)"]]

# Sütun sırasını değiştirdikten sonra sonucu yazdır
df_log.head(10)
#        Actual  Predicted  Actual_original  Predicted_original  Prediction Accuracy (%)
# 5812    6.019      5.900          410.000             364.129                   98.034
# 5061    5.775      5.871          321.000             353.488                   98.335
# 9955    5.124      5.164          167.000             173.778                   99.228
# 923     2.708      2.978           14.000              18.641                   90.046
# 10939   4.920      4.917          136.000             135.566                   99.936
# 16948   4.796      4.507          120.000              89.674                   93.984
# 9889    1.946      1.659            6.000               4.255                   85.262
# 17277   5.288      5.187          197.000             177.921                   98.084
# 5703    1.946      1.993            6.000               6.334                   97.604
# 14511   5.338      5.514          207.000             247.246                   96.686

df_log["Prediction Accuracy (%)"].mean()  # 95.877 ortalama tahmin doğruluğu (%)


#########################
# Kaggle'da Submit Etmek:
#########################

# Orijinal ölçekteki tahminleri içeren bir DataFrame oluşturuyoruz
submit_df = pd.DataFrame({
    "Predicted": df_log['Predicted_original']})    # Log dönüşümünden çıkarılmış orijinal tahminler

# DataFrame'i CSV formatında dışa aktar (index olmadan)
submit_df.to_csv("submission_bike_sharing_model.csv", index=False)

print("Tahminler için submission.csv dosyası oluşturuldu.")


#########################
# Veri Setinden Ratgele Örneklem Çekip, Tahminlere Bakmak:
#########################

# 1. Rastgele 3 örnek seç
sample_df = X_test.sample(n=3)

# 2. Seçilen rastgele 3  örnekle tahmin yapma
y_pred_sample = final_model.predict(sample_df)
# array([5.08968651, 5.97855495, 5.94415324])

# 3. Gerçek değerler ile tahmin edilen değerleri karşılaştırma

# Gerçek değerler için y_test'ten bu örneklerin karşılıklarını alalım
y_actual_sample = y_test.loc[sample_df.index]
# 6271    4.934
# 10617   6.157
# 4478    5.935

# 4. Sonuçları birleştirelim
comparison_sample_df = pd.DataFrame({
    'Actual': y_actual_sample,  # Loglu değerler
    'Actual_original': np.expm1(y_actual_sample),  # Gerçek (orijinal) değerler
    'Predicted': y_pred_sample,  # Tahmin edilen loglu değerler
    'Predicted_original': np.expm1(y_pred_sample),  # Tahmin edilen orijinal değerler
    'Prediction Accuracy (%)': (1 - abs(y_pred_sample - y_actual_sample) / y_actual_sample) * 100})  # Başarı yüzdesi

# 5. Sonuçları yazdır
print("Üç örnek için Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması:")
print(comparison_sample_df)

# Üç örnek için Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması:
#        Actual  Actual_original  Predicted  Predicted_original  Prediction Accuracy (%)
# 6271    4.934          138.000      5.090             161.339                   96.855
# 10617   6.157          471.000      5.979             393.869                   97.102
# 4478    5.935          377.000      5.944             380.516                   99.844







