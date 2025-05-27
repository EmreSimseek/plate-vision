# PlateVision - Araç Kapısı Plaka Tespit ve Otomasyon Sistemi

PlateVision, kapı önüne yanaşan araçların türünü ve plakalarını tespit ederek, önceden tanımlanmış yetkili plakalara göre otomatik kapı açma/kapama veya uyarı verme işlemlerini gerçekleştiren bir web uygulamasıdır. Sistem, özellikle kamyon ve TIR gibi büyük araçların kontrollü geçişini sağlamak amacıyla geliştirilmiştir.

## Projenin Amacı

Bu projenin temel amacı, bir tesise veya alana araç giriş-çıkışlarını otomatikleştirmek, güvenliği artırmak ve manuel kontrol ihtiyacını azaltmaktır. Sistem, gerçek zamanlı kamera görüntülerini işleyerek araç tespiti, plaka okuma ve veritabanı karşılaştırması yapar.

## Temel Özellikler

*   **Gerçek Zamanlı Araç Tespiti:** Kamera görüntüsünden kamyon, TIR ve diğer araç türlerini tespit eder (YOLOv8 tabanlı).
*   **Plaka Okuma (OCR):** Tespit edilen araçların plakalarını otomatik olarak okur (EasyOCR tabanlı).
*   **Veritabanı Entegrasyonu:**
    *   Yetkili plakaların kaydedilmesi, listelenmesi ve yönetilmesi.
    *   Tespit edilen plakaların veritabanındaki kayıtlarla karşılaştırılması.
*   **Otomasyon Mantığı:**
    *   Eğer yanaşan araç **kamyon/TIR** ise ve plakası veritabanında **yetkili** olarak kayıtlıysa, sistem (simüle edilmiş) bir kapıyı açar.
    *   Eğer yanaşan araç **kamyon/TIR değilse** (örn: otomobil) veya plakası **yetkisiz/kayıtsız** ise, sistem bir uyarı (sesli/görsel) verir.
*   **Web Arayüzü (Flask):**
    *   **Canlı İzleme:** Anlık kamera görüntüsü, tespit edilen nesneler (araç, plaka) ve sistem durumu mesajları gösterilir.
    *   **Plaka Düzeltme:** Görevlinin, OCR tarafından yanlış okunan kamyon plakalarını anlık olarak düzeltebilmesi için arayüz sunar.
    *   **Plaka Yönetimi:** Yetkili plakaların eklenebildiği, listelenebildiği ve silinebildiği bir yönetim sayfası.
    *   **Aktivite Logları:** Tüm araç tespitleri, plaka okuma denemeleri, yetkilendirme sonuçları ve kullanıcı düzeltmeleri zaman damgası ve görsellerle birlikte kaydedilir ve listelenir.
    *   **Ayarlar:** Tespit yapılacak Bölgesel İlgi Alanı (ROI) gibi sistem ayarlarının yapılandırılabildiği bir sayfa.
*   **Bölgesel İlgi Alanı (ROI):** Tespitlerin sadece videonun belirli bir bölgesinde yapılması için kullanıcı tarafından ayarlanabilir ROI.
*   **Takip Sistemi:** Aynı aracın defalarca işlenmesini engellemek ve olayları daha iyi yönetmek için temel bir nesne takip mekanizması.
*   **Görsel Kayıt:**
    *   Başarılı/başarısız tüm önemli araç geçiş denemelerine ait araç görüntüleri kaydedilir.
    *   OCR işlemine giren ham plaka kesitleri de analiz amacıyla kaydedilebilir.

## Kullanılan Teknolojiler

*   **Backend:** Python, Flask
*   **Nesne Tespiti:** YOLOv8 (Ultralytics kütüphanesi)
*   **Plaka Okuma (OCR):** EasyOCR
*   **Veritabanı:** SQLite
*   **Frontend:** HTML, CSS, JavaScript
*   **Görüntü İşleme:** OpenCV
