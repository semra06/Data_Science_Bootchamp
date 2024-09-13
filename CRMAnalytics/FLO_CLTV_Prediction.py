##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


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

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: '%.4f' % x)

# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
           df_ = pd.read_csv("Week 3 - CRM Analytics/FLOMusteriSegmentasyonu/flo_data_20k.csv")
           df = df_.copy()

           # 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
            # Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
            def outlier_thresholds(dataframe, variable):
                quartile1 = dataframe[variable].quantile(0.01)
                quartile3 = dataframe[variable].quantile(0.99)
                interquantile_range = quartile3 - quartile1
                up_limit = quartile3 + 1.5 * interquantile_range
                low_limit = quartile1 - 1.5 * interquantile_range
                return round(low_limit), round(up_limit)

            l, u = outlier_thresholds(df, 'customer_value_total_ever_online')

            def replace_with_thresholds(dataframe, variable):
                low_limit, up_limit = outlier_thresholds(dataframe, variable)
                dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
                dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

           # 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
           # aykırı değerleri varsa baskılayanız.
            df.describe().T
            replace_with_thresholds(df, "order_num_total_ever_online")
            replace_with_thresholds(df, "order_num_total_ever_offline")
            replace_with_thresholds(df, "customer_value_total_ever_offline")
            replace_with_thresholds(df, "customer_value_total_ever_online")

            # ya da

            replace_th_list = ["order_num_total_ever_online", "order_num_total_ever_offline",
                               "customer_value_total_ever_offline", "customer_value_total_ever_online"]
            for var in replace_th_list:
                replace_with_thresholds(df, var)


           # 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
           # alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
           df.head()
           df["order_number_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
           df["customer_value_total"] =df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

           # 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
           for col in df.columns:
               if "date" in col:
                   df[col] = pd.to_datetime(df[col])

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
           # 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
           df["last_order_date"].max()
           today_date = dt.datetime(2021, 6, 1)

           # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
           # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
           cltv = pd.DataFrame()
           cltv["customer_id"] = df["master_id"]

           cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"])
           cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"].apply(lambda x: x.days) / 7

           cltv["T_weekly"] = df["first_order_date"].apply(lambda x: (today_date - x).days) / 7

           cltv["frequency"] = df["order_number_total"]
           cltv = cltv[cltv["frequency"] > 1]

           cltv["monetary_cltv_avg"] = df["customer_value_total"] / df["order_number_total"]


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
           # 1. BG/NBD modelini fit ediniz.
           bgf = BetaGeoFitter(penalizer_coef=0.001)
           bgf.fit(cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

                # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
                bgf.predict(12, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])
                cltv["exp_sales_3_month"] = bgf.predict(12, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

                # tahmin sonuçlarını değerlendirmek için aşağıdaki grafiği inceleyebiliriz
                plot_period_transactions(bgf)
                plt.show()

                # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
                bgf.predict(24, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])
                cltv["exp_sales_6_month"] = bgf.predict(24, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

           # 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
           ggf = GammaGammaFitter(penalizer_coef=0.01)
           ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

           ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])
           cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])

           # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
            cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                                           cltv["frequency"],
                                                           cltv["recency_cltv_weekly"],
                                                           cltv["T_weekly"],
                                                           cltv["monetary_cltv_avg"],
                                                           time=6,  # 6 aylık
                                                           freq='W',  # T'nin frekans bilgisi yani haftalık
                                                           discount_rate=0.01)


                # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
                cltv.sort_values(by="cltv", ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
           cltv["cltv_segment"] = pd.qcut(x=cltv["cltv"], q=4, labels=["D", "C", "B", "A"])

           # 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
           cltv.groupby("cltv_segment").agg({"count", "mean", "sum"})

# BONUS: Tüm süreci fonksiyonlaştırınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit.round(), up_limit.round()


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def create_cltv_pred(dataframe, month=6):

    # aykırı değerlerin baskılanması
    replace_with_thresholds(dataframe, "order_num_total_ever_online")
    replace_with_thresholds(dataframe, "order_num_total_ever_offline")
    replace_with_thresholds(dataframe, "customer_value_total_ever_offline")
    replace_with_thresholds(dataframe, "customer_value_total_ever_online")

    # toplam sipariş sayısının ve toplam sipariş harcamalarının hesaplanması
    dataframe["order_number_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    # zaman belirten değişkenlerin tiplerinin "date" tipine dönüştürülmesi
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    # analiz yapılacak tarihin belirlenmesi
    today_date = dt.datetime(2021, 6, 1)

    # cltv için gerekli değerlerin hesaplanması
    cltv = pd.DataFrame()
    cltv["customer_id"] = dataframe["master_id"]

    cltv["recency_cltv_weekly"] = (dataframe["last_order_date"] - dataframe["first_order_date"])
    cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"].apply(lambda x: x.days) / 7

    cltv["T_weekly"] = dataframe["first_order_date"].apply(lambda x: (today_date - x).days) / 7

    cltv["frequency"] = dataframe["order_number_total"]
    cltv = cltv[cltv["frequency"] > 1]

    cltv["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_number_total"]

    # BG/NBD modelinin kurulumu
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

    # 'month' aylık periyotta beklenen satış sayısı hesaplanması ve dataframe'e eklenmesi
    cltv["exp_sales_6_month"] = bgf.predict(month * 4, cltv["frequency"], cltv["recency_cltv_weekly"], cltv["T_weekly"])

    # Gamma-Gamma modelinin kurulumu
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

    # beklenen ortalama satış geliri hesaplaması ve dataframe'e eklenmesi
    cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])

    # 'month' aylık cltv hesaplanması ve dataframe'e eklenmesi
    cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                               cltv["frequency"],
                                               cltv["recency_cltv_weekly"],
                                               cltv["T_weekly"],
                                               cltv["monetary_cltv_avg"],
                                               time=month,  # 'month' aylık
                                               freq='W',  # T'nin frekans bilgisi yani haftalık
                                               discount_rate=0.01)

    # müşterilen segmentlere ayrılması
    cltv["cltv_segment"] = pd.qcut(x=cltv["cltv"], q=4, labels=["D", "C", "B", "A"])

    return cltv
