import pandas as pd
import math
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# N, bir reklamı test ederek göstereceğimiz kullanıcı sayısıdır.
# toplamda 10.000 satır var, her biri bir kullanıcıyı temsil ediyor, 
# bu nedenle bu sayı 1 ile 10.0000 arasında herhangi bir yerde olabilir
N = 10000
# d, karar vermeye çalıştığımız reklam sayısıdır
# 10 sütun var, her biri farklı bir reklamı temsil ediyor, 
# bu yüzden 10'a eşitledik çünkü 10'dan hangisinin en iyi olduğunu görmek istiyoruz
d = 10
# Bu liste seçtiğimiz tüm reklamları içerecektir. Bu listenin uzunluğu N ile aynı olacaktır.
# NOT: Bu, *TIKLANAN* bir reklamı temsil etmez. Sadece bu reklamın bir kullanıcıya gösterilmesini seçtiğimizi söylüyor. 
# Kullanıcı, göstermek üzere seçtiğimiz bu reklamlardan herhangi birine tıklayabilir veya tıklamayabilir
ads_selected = []
# Bu liste, her reklam için, onları kaç kez seçtiğimizi temsil eden bir değer içerir.
# Not: Tekrar ediyorum, bu tıklamaları temsil etmiyor. 
# Sadece her reklamı bir kullanıcıya kaç kez göstermeyi seçtiğimizi takip ediyor.
numbers_of_selections = [0] * d
# Bu liste her reklamın kaç kez tıklandığını içerir.
# Yani kullanıcı n'e gösterilecek bir reklam seçtik ve o reklama tıkladılar. 
# Bu liste bunun kaç kez gerçekleştiğini izlemek için kullanılacak
sums_of_rewards = [0] * d
# Kullanıcıların kendilerine gösterdiğimiz herhangi bir reklama kaç kez tıkladıklarının toplam sayısı.
# Bu, sum(sum_of_rewards) ile aynı olacaktır.
total_reward = 0
#n, reklam göstereceğimiz her kullanıcıyı temsil eder.
for n in range(0,N):
    # reklam değişkeni hangi reklamı gösterdiğimizi izlemek için kullanılacak
    a = 0
    # max_upper_bound, şu anda en yüksek üst sınıra sahip olan 10 reklamdan hangisinin olduğunu takip etme yöntemimizdir
    max_upper_bound = 0
    # Bu döngüde, 10 reklamımızın her birini dolaşacağız ve böylece kullanıcımız 'n'e gösterilecek 10 reklamdan 1'ini seçebileceğiz
    # bu for döngüsünün amacı hangi reklamın en yüksek üst sınıra sahip olduğunu bulmaktır
    for i in range(0,d):
        #reklam bir kullanıcıya en az bir kez gösterilmişse:
        if (numbers_of_selections[i] > 0):
            # average_reward (reklamın tıklanma sayısı) / (reklamın gösterilme sayısı)
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            # delta_i videoda gösterilen formülden hesaplanır. Güven aralığımızı temsil eder.
            # n+1 şu ana kadar gösterilen toplam reklam sayısıdır. number_of_selections[i] reklam i'nin toplam kaç kez gösterildiğidir
            delta_i= math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            # bu üst güven sınırıdır. Reklamın ortalama tıklama oranımızı (ortalama_ödül) alır ve güven aralığımızı ekler.
            # bu bir tür belirsizlik gibi. Ortalama tıklama oranının ortalama ödül olduğunu düşünüyoruz, 
            # ancak *ortalama* ödül + delta_i kadar büyük olabilir
            upper_bound = average_reward + delta_i
        else:
            # bu 'else' durumunda reklam henüz bir kullanıcıya gösterilmedi, bu yüzden üst sınırını çok büyük bir sayıya ayarladık
            # bu, her reklamı mümkün olduğunca çabuk test ettiğimizden emin olmamızı sağlayacaktır
            upper_bound = 1e400
        # 10 reklamımız arasında döngü oluştururken, 10 reklamımızdan hangisinin en yüksek üst_sınıra sahip olduğunu izlemek için bu max_upper_bound'u kullanırız
        if (upper_bound > max_upper_bound):
            # eğer upper_bound > max_upper_bound ise, bu upper_bound'u yeni maksimum upper_bound olarak atamamız gerekir
            max_upper_bound = upper_bound
            # Hangi reklamın maksimum üst sınıra sahip olduğunu takip etmemiz gerekiyor, bu yüzden onu "ad" değişkeniyle takip ediyoruz
            ad = i
            
     # bu for döngüsü 10 reklamımız arasında dolaştıktan sonra, en yüksek üst_sınıra sahip olan reklam ad = i'ye atanacaktır
    # yani ilk reklamımızın en yüksek üst sınırı varsa, ad = 0
 
    # BU, KULLANICIYA GÖSTERECEĞİMİZ SEÇİLMİŞ REKLAMDIR
  # seçtiğimiz reklamı, seçtiğimiz tüm reklamları takip eden listemize ekliyoruz
    ads_selected.append(ad)
    # Her reklamın toplam sayısını izleyen listemize 1 ekliyoruz
    numbers_of_selections[ad] += 1
    # şimdi gerçeğin zamanı. İşte bu noktada veri setimize göz atıp şu soruyu yanıtlıyoruz:
    # KULLANICI, KENDİSİNE GÖSTERMEYİ SEÇTİĞİMİZ REKLAMI TIKLAYACAK MI?
    # sütun ad'in (reklam numaramız) n. satırının (kullanıcı numarası) 1 mi yoksa 0 mı olduğunu kontrol ediyoruz
    # 1 kullanıcının reklamlarımıza tıkladığı anlamına gelir, 0 ise tıklamadığı anlamına gelir
    reward = dataset.values[n, ad]
    # sonra bu ödülü, her reklam için toplam tıklama sayısını izleyen listemize ekliyoruz
    sums_of_rewards[ad] += reward
    # sonra bu ödülü, tüm reklamların toplam tıklama sayısını izleyen int'e ekliyoruz
    total_reward += reward


plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()