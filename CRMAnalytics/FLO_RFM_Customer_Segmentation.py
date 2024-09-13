import datashader.examples.streaming
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("Week 3 - CRM Analytics/FLOMusteriSegmentasyonu/flo_data_20k.csv")
           # 2. "Veri setinde
                     # a. İlk 10 gözlem,
                        df.head(10)
                     # b. Değişken isimleri,
                        df.columns
                     # c. Betimsel istatistik,
                        df.describe().T
                     # d. Boş değer,
                        df.isnull().sum()
                     # e. Değişken tipleri, incelemesi yapınız.
                        df.dtypes
           # 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
            df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
            df["customer_total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

           # 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
            df.dtypes
            df["first_order_date"] = pd.to_datetime(df["first_order_date"])
            df["last_order_date"] = pd.to_datetime(df["last_order_date"])
            df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
            df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

            # ya da

            for col in df.columns:
                if "date" in col:
                    df[col] = pd.to_datetime(df[col])

           # 5. Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
            df.groupby("order_channel").agg({"master_id": "count",
                                             "order_num_total": "sum",
                                             "customer_total_value": "sum"})

           # 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
            df["customer_total_value"].sort_values(ascending=False).head(10)
           # 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
            df["order_num_total"].sort_values(ascending=False).head(10)
           # 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
            def data_prep(df_):
                df_["order_num_total"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
                df_["customer_total_value"] = df_["customer_value_total_ever_online"] + df_["customer_value_total_ever_offline"]

                for col in df_.columns:
                    if "date" in col:
                        df_[col] = pd.to_datetime(df_[col])

                return df_

            new_df = data_prep(df)
            new_df.dtypes

# GÖREV 2: RFM Metriklerinin Hesaplanması

    # 1. Recency, Frequency, Menoetary tanımlarını yapınız
    # Recency = ????
    # Frequency = ????
    # Monetary = ????

    # 2. Müşteri özelinde RFM metriklerini hesaplayınız.
    df["last_order_date"].max()
    today_date = dt.datetime(2021, 6, 1)

    df["Recency"] = [(today_date - date).days for date in df["last_order_date"]]
    df["Frequency"] = df["order_num_total"]
    df["Monetary"] = df["customer_total_value"]

    # 3. Hesapladığınız metrikleri rfm isimli bir değişkene atayınız
    rfm = df[["master_id", "Recency", "Frequency", "Monetary"]]


# GÖREV 3: RF ve RFM Skorlarının Hesaplanması

    # 1. Recency, Frequency, Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
    rfm["recency_score"] = pd.qcut(x=rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(x=rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(x=rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

# Örnek: method="first"
#        11,11,11,11,22,23,32,32,45,45,59,59,60,61,62
#        11,11,11 | 11,22,23,| 32,32,45 | 45,59,59 | 60,61,62
#            1    |    2     |     3    |    4     |     5

# Örnek: method="dense"
#        11,11,11,11,22,23,32,32,45,45,59,59,60,61,62
#        11,11,11,11 | 22,23,| 32,32,45 | 45,59,59 | 60,61,62
#            1       |   2   |     3    |    4     |     5

# Örnek: method="min"
#        11,11,11,11,22,23,32,32,45,45,59,59,60,61,62
#        11,11,11,11 | 22,23,32 | 32,45,45 | 59,59,60 |61,62
#              1     |    2     |     3    |    4     |  5

    # 3. recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)


# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması

    # 1. Oluşturulan RF skorları için segment tanımlamaları yapınız.
    seg_map = {
        r'[1-2][1-2]': 'hibernating',  # birinci ve ikinci elemanında 1 ya da 2 görürsen 'hibernating' diye isimlendir
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',  # birinci ve ikini elemanı 3 ise 'need_attention' diye isimlendir
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    # 2. Yukarıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# SettingWithCopyWarning uyarılarını kapatmak için
pd.options.mode.chained_assignment = None  # default='warn'

# GÖREV 5: Aksiyon zamanı!
           # 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
            rfm.groupby("segment").agg({"Recency": "mean",
                                        "Frequency": "mean",
                                        "Monetary": "mean"})

           # 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
                   # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
                   # tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers),
                   # ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına
                   # yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.

                    #  rfm'e ilgili kategorileri "category" isimli bir sütun olarak atıyoruz:
                    rfm["category"] = df["interested_in_categories_12"]

                    # rfm'in indexini "mister_id" olarak atıyoruz:
                    rfm = rfm.set_index("master_id")

                    # kategorisinde "KADIN" olan ve segmenti champions ve loyal_customers olan indexleri (yani id'leri) seçeceğiz
                    new_df = pd.DataFrame()
                    new_df["yeni_marka_hedef_müşteri_id"] = rfm[((rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")) & rfm["category"].str.contains("KADIN")].index

                    # bu dataframe'i bir csv dosyası olarak kaydediyoruz
                    new_df.to_csv("yeni_marka_hedef_müşteri_id.csv")



                   # b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir
                   # alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor.
                    # Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv olarak kaydediniz.

                    # segmenti "cant_loose", "hibernating" ve "new_customer" olan müşterilerin id'lerini bulacağız
                    new_df_2 = pd.DataFrame()
                    new_df_2["indirim_hedef_müşteri_ids"] = rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "hibernating") | (rfm["segment"] == "new_customers")].index

                    # bu dataframe'i bir csv dosyası olarak kaydediyoruz
                    new_df_2.to_csv("indirim_hedef_müşteri_ids.csv")


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.

def create_rfm(df_):

    # Veriyi önhazırlama
        # Toplam sipariş sayısını ve toplam harcalama değişkenlerini hesaplama
    df_["order_num_total"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
    df_["customer_total_value"] = df_["customer_value_total_ever_online"] + df_["customer_value_total_ever_offline"]

        # Tarih belirlen girdilerin tipini date'e çevirme
    for col in df_.columns:
        if "date" in col:
            df_[col] = pd.to_datetime(df_[col])

    # RFM metriklerini hesaplama
    today_date = dt.datetime(2021, 6, 1)
    df_["Recency"] = [(today_date - date).days for date in df_["last_order_date"]]
    df_["Frequency"] = df_["order_num_total"]
    df_["Monetary"] = df_["customer_total_value"]

        # Hazırladğımız metriklerin rfm isimli bir dataframe'e aktarılması
    rfm = df_[["master_id", "Recency", "Frequency", "Monetary"]]

    # RFM skorlarınının hesaplanması
    rfm["recency_score"] = pd.qcut(x=rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(x=rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(x=rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5])

    # RF skorunun oluşturulması
    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    # Segmenlerin RF skoruna göre oluşturulması
    seg_map = {
        r'[1-2][1-2]': 'hibernating',  # birinci ve ikinci elemanında 1 ya da 2 görürsen 'hibernating' diye isimlendir
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',  # birinci ve ikini elemanı 3 ise 'need_attention' diye isimlendir
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

    # indexlerin id'ler olarak ayarlanması
    rfm = rfm.set_index("master_id")

    return rfm

new_rfm = create_rfm(df)