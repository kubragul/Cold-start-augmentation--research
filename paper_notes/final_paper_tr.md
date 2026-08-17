# Soğuk Başlangıç Finans Tahmini için Sentetik Zaman Serisi Artırma: Pilot Çalışma

## Özet

Bu çalışma, kısa gözlem geçmişine sahip finansal serilerde (cold-start) istatistiksel sentetik artırmanın tahmin başarısını iyileştirip iyileştirmediğini test eder. 2020-2024 döneminde 11 büyük ölçekli ABD hissesi (4 sektör) üzerinde 1,419 yuvarlanan senaryo üretilmiştir. Eğitim pencereleri 4, 8 ve 12 hafta; tahmin ufku 28 işlem günüdür. Baseline olarak naive, moving-average ve linear-trend modelleri kullanılmıştır. Artırma yöntemi, eğitim penceresinde doğrusal trend + Gauss artık gürültüsü varsayımıyla sentetik devam noktaları üretip seriye ekler. Sonuçlar, bu yaklaşımın genel performansı anlamlı biçimde kötüleştirdiğini göstermiştir: ortalama MAPE 6.4355'ten 16.0382'ye çıkmıştır; ortalama eşli fark +9.6027 MAPE puanıdır. Birincil Wilcoxon signed-rank testi (ve FDR-BH düzeltilmiş sonuçlar) "significantly_worsened" sonucunu vermiştir.

## 1. Giriş

Cold-start tahminde temel zorluk, modelin hedef seriye ait çok az geçmiş veriyle kısa vadeli öngörü üretmesidir. Sentetik veri artırma, veri miktarını yükselttiği için cazip görünse de, tahmin için kritik olan yerel yapı ve uç nokta bilgisini bozma riski taşır. Bu çalışma, "daha fazla veri" varsayımını değil, "daha doğru yapıyı koruma" varsayımını sınar.

## 2. Araştırma Sorusu ve Hipotez

**Araştırma sorusu:** İstatistiksel sentetik artırma, finans cold-start tahmininde doğruluğu artırır mı?

**Hipotez:** Sentetik noktalar gerçek serinin tahmin edici yapısını korursa fayda sağlayabilir; aksi halde özellikle uç nokta duyarlı modellerde hata artışı görülebilir.

## 3. Veri ve Deney Tasarımı

- **Kaynak:** `yfinance` günlük adjusted close fiyatları
- **Varlıklar:** 11 hisse (Technology, Finance, Consumer, Energy)
- **Dönem:** 2020-01-01 - 2024-12-31
- **Senaryo üretimi:** Yuvarlanan pencere yaklaşımı
- **Eğitim pencereleri:** 4/8/12 hafta (yaklaşık 20/40/60 işlem günü)
- **Tahmin ufku:** 28 işlem günü
- **Adım:** 28 işlem gözlemi
- **Toplam senaryo:** 1,419

## 4. Yöntem

### 4.1 Baseline Modeller

- **Naive:** Son gözlemi tekrar eder
- **Moving average:** Son 5 gözlemin ortalamasını tekrar eder
- **Linear trend:** Eğitim penceresine OLS trend uydurup ileri uzatır

### 4.2 İstatistiksel Artırma

Her eğitim serisi için:
1. Doğrusal trend uydurulur
2. Artıklar hesaplanır
3. Artık standart sapması tahmin edilir
4. Trend ileri uzatılarak sentetik devam noktaları üretilir
5. Gauss gürültüsü eklenir
6. Sentetik noktalar eğitim sonuna eklenir

Artırma oranları: 0.5x, 1.0x, 2.0x

### 4.3 Değerlendirme ve İstatistik

- Metrikler: MAE, RMSE, MAPE
- Eşli karşılaştırma: aynı sample-model için baseline vs augmented
- Testler: Wilcoxon (birincil), paired t-test (destekleyici)
- Çoklu test düzeltmesi: Benjamini-Hochberg FDR

## 5. Bulgular

- **Baseline ortalama MAPE:** 6.4355
- **Augmented ortalama MAPE:** 16.0382
- **Ortalama eşli fark (MAPE):** +9.6027
- **İyileşen karşılaştırma oranı:** %20.91 (2,670 / 12,771)
- **En az zararlı oran:** 0.5
- **En zararlı oran:** 2.0
- **Genel Wilcoxon yorumu:** significantly_worsened
- **FDR-BH sonrası yorum:** significantly_worsened

Sonuç, artırılan sentetik blok büyüdükçe hata bozulmasının arttığını göstermektedir.

## 6. Tartışma

Muhtemel hata mekanizması "uç nokta bozulmasıdır". Sentetik devam noktaları eğitim sonuna eklendiğinde, özellikle naive ve moving-average modellerinin dayandığı son gözlem sinyali gerçek piyasadan kopar. Linear-trend modelinde ise trend üzerine trend etkisi oluşarak yapay eğilim büyütülebilir.

## 7. Kısıtlar

- Finans odaklı, sınırlı varlık seti (11 hisse)
- Basit baseline modeller
- Fiyat seviyesinde modelleme (getiri tabanlı değil)
- Yuvarlanan pencereler arası örtüşme nedeniyle bağımsızlık varsayımı zayıflayabilir

## 8. Sonuç

Bu pilotta kullanılan doğrusal-trend + Gauss artık temelli sentetik devam artırması, cold-start finans tahmininde fayda sağlamamış; tersine performansı anlamlı biçimde kötüleştirmiştir. Bulgular, sentetik veri miktarından çok sentetik verinin hedef serinin tahmin edici yapısını ne ölçüde koruduğunun kritik olduğunu göstermektedir.

## 9. Şekiller

Şekiller `results/figures_revised/` dizininde üretilmiştir:

- `baseline_mape_by_model_window.png`
- `augmentation_mape_by_ratio_window.png`
- `mape_difference_heatmap.png`
- `improvement_rate_by_model_ratio.png`
- `sector_level_mape_difference.png`
- `baseline_vs_augmented_distribution.png`
- `baseline_vs_augmented_distribution_zoomed.png`
- `example_forecast_failure_case.png`
- `example_forecast_improvement_case.png`

