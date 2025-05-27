# PlateVision/core/ocr_module.py
import sys
import os
import cv2
import numpy as np
import easyocr
from typing import Optional, List, Tuple
import re # Regex ile plaka formatı kontrolü için

# --- Python PATH Ayarı ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- END Python PATH Ayarı ---
import config

ocr_reader: Optional[easyocr.Reader] = None
turkish_plate_chars = '0123456789ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ' # Tanınacak karakterler

def load_ocr_model() -> bool:
    global ocr_reader
    languages = config.OCR_LANGUAGES
    if not languages:
        print("HATA: OCR için config'de dil belirtilmemiş."); return False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"EasyOCR için GPU kullanılabilirliği: {use_gpu}")
        # Model yükleme sırasında bazı parametreler denenebilir, örn: recognizer, model_storage_directory
        ocr_reader = easyocr.Reader(languages, gpu=use_gpu, recognizer='standard') # 'standard' veya 'custom' (eğitilmişse)
        print(f"EasyOCR modeli yüklendi. Diller: {languages}. GPU: {use_gpu}")
        return True
    except ImportError: print("HATA: 'easyocr' veya 'torch' kurulu değil."); return False
    except Exception as e: print(f"HATA: EasyOCR yüklenirken: {e}"); return False

def preprocess_plate_roi_for_ocr(plate_image_roi: np.ndarray) -> np.ndarray:
    """OCR doğruluğunu artırmak için plaka görüntüsünü ön işler."""
    if plate_image_roi is None or plate_image_roi.size == 0: return np.array([])
    
    img = plate_image_roi.copy()

    # 1. Gri Tonlama
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Yeniden Boyutlandırma (Karakter yüksekliğini optimize etmek için)
    # OCR modelleri belirli bir karakter yüksekliğinde daha iyi çalışabilir.
    # Örnek: Hedef yükseklik 32-64 piksel arası.
    # En/boy oranını koruyarak yeniden boyutlandırma.
    # target_h = 48
    # h, w = gray.shape
    # if h > 0 and w > 0:
    #     scale = target_h / h
    #     new_w, new_h = int(w * scale), target_h
    #     if new_w > 0:
    #         gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC) # Veya INTER_LANCZOS4

    # 3. Gürültü Azaltma
    # GaussianBlur küçük gürültüler için, medianBlur tuz-biber gürültüsü için.
    # kernel_size = (3,3) # veya (5,5)
    # blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    # Alternatif: Bilateral filter kenarları korurken gürültüyü azaltır ama yavaştır.
    # blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # Şimdilik hafif bir blur
    blurred = cv2.GaussianBlur(gray, (3,3), 0)


    # 4. Kontrast Artırma (CLAHE)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Daha genel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4)) # Daha lokal, küçük bölgeler için
    enhanced_contrast = clahe.apply(blurred) # veya gray'e uygula

    # 5. Binarizasyon (Eşikleme) - Bu adım çok kritiktir ve duruma göre ayarlanmalı
    # EasyOCR kendi içinde bir binarizasyon yapabilir. Manuel binarizasyon bazen
    # daha iyi sonuç verir, bazen kötüleştirir. Deneyerek karar verin.
    # Otsu's binarization genellikle iyi bir başlangıçtır.
    # _, binary_image = cv2.threshold(enhanced_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Alternatif: Adaptif Eşikleme (Değişken aydınlatma koşulları için daha iyi olabilir)
    # binary_image = cv2.adaptiveThreshold(enhanced_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                     cv2.THRESH_BINARY_INV, blockSize=11, C=5) # INV, karakterler beyaz, arka plan siyah
                                                                                # EasyOCR genellikle koyu karakter/açık arka plan bekler.
                                                                                # Bu yüzden THRESH_BINARY veya THRESH_BINARY_INV denenebilir.
    # Şimdilik binarizasyon uygulamadan, kontrastı artırılmış gri görüntüyü kullanalım.
    # EasyOCR'a bu görüntüyü verelim, kendi iç mekanizmalarını kullansın.
    final_image_for_ocr = enhanced_contrast


    # 6. Opsiyonel: Morfolojik Operasyonlar (Karakterleri netleştirmek için)
    kernel = np.ones((2,2),np.uint8) # Veya cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    eroded = cv2.erode(final_image_for_ocr, kernel, iterations = 1)
    dilated = cv2.dilate(final_image_for_ocr, kernel, iterations = 1)
    opening = cv2.morphologyEx(final_image_for_ocr, cv2.MORPH_OPEN, kernel) # Gürültü temizleme
    closing = cv2.morphologyEx(final_image_for_ocr, cv2.MORPH_CLOSE, kernel) # Küçük delikleri kapatma
    final_image_for_ocr = opening # Veya closing, veya ikisi birden

    # 7. Opsiyonel: Kenar Yumuşatma / Kenar Belirginleştirme (Çok dikkatli kullanılmalı)
    # Bazen karakter kenarlarını çok hafif bulanıklaştırmak veya tam tersi keskinleştirmek işe yarayabilir.

    # DEBUG: Ön işlenmiş görüntüyü göster
    # cv2.imshow("Preprocessed for OCR", final_image_for_ocr)
    # cv2.waitKey(1)
    
    return final_image_for_ocr

def postprocess_ocr_text(raw_text: str) -> str:
    """OCR'dan gelen ham metni temizler ve plaka formatına uydurmaya çalışır."""
    if not raw_text: return ""

    # 1. Sadece izin verilen karakterleri al ve büyük harf yap
    # (EasyOCR allowlist ile zaten bunu büyük ölçüde yapmalı)
    text_with_allowed_chars = ''.join(char for char in raw_text.upper() if char in turkish_plate_chars)

    # 2. Yaygın OCR hatalarını düzeltme (Örnekler, daha fazlası eklenebilir)
    # Bu düzeltmeler çok dikkatli yapılmalı, doğru karakterleri bozabilir.
    # Öncelikle en bariz ve güvenilir olanları ekleyin.
    corrected_text = text_with_allowed_chars
    # corrected_text = corrected_text.replace('O', '0') # '0' mı 'O' mu ayrımı zor
    # corrected_text = corrected_text.replace('I', '1') # '1' mi 'I' mı
    # corrected_text = corrected_text.replace('S', '5')
    # corrected_text = corrected_text.replace('B', '8')
    # corrected_text = corrected_text.replace('G', '6')
    # corrected_text = corrected_text.replace('Z', '2')
    # ... vb. Bu tür düzeltmeler için bir sözlük kullanılabilir.

    # 3. Plaka formatına göre desen kontrolü (Regex)
    # Bu, en güçlü filtrelerden biridir.
    # Örnek Türk Plakası Desenleri:
    # NN LL NNN, NN LLL NN, NN L NNNN vb. (N: Rakam, L: Harf)
    # Basit bir regex (daha karmaşık ve kapsamlı olabilir):
    # ^[0-8][0-9] -> İl kodu (01-81)
    # [A-ZÇĞİÖŞÜ]{1,3} -> 1-3 Harf
    # [0-9]{2,4} -> 2-4 Rakam
    # Bu regex, harf ve rakam grupları arasında boşluk olmadığını varsayar.
    # Boşluklu formatlar için regex'i güncellemek gerekir.
    # Şu anki cleaned_text boşluksuz olduğu için bu regex uygun olabilir.
    
    # Daha genel bir regex: İl kodu (2 rakam), sonra harf grubu (1-3 harf), sonra rakam grubu (2, 3 veya 4 rakam)
    # Bu regex, plakanın standart kısımlarını ayırmaya çalışır.
    # Pattern1: 01 L 1234, 01 LL 123, 01 LLL 12
    # Pattern2: 01 L 123, 01 LL 12 (Eski tip ilçe plakaları için pek uygun değil)
    # Pattern3: 01 L 12 (Motosiklet vb.)

    # Sadece karakter yapısını kontrol eden bir regex (Türkçe harfler dahil)
    # Bu, "06ABC06" gibi bir sonucu doğrular.
    # Daha karmaşık formatlar (örn: geçici plakalar, resmi plakalar) için regex'ler özelleştirilmeli.
    # Şimdilik, sadece alfanümerik ve belirli uzunlukta olmasını kontrol edelim.
    # (Bu, `read_plate_text` içindeki son filtrelemeye benziyor, burada daha detaylı olabilir)

    # Örnek bir plaka desenini zorlamak: "NN LLL NN" veya "NN LL NNN"
    match = re.fullmatch(r"([0-8][0-9])([A-Z]{2,3})([0-9]{2,4})", corrected_text)
    if match:
        return "".join(match.groups()) # Eşleşen grupları birleştir

    # Şimdilik sadece uzunluk kontrolü yapalım, regex daha sonra eklenebilir.
    if len(corrected_text) >= 5 and len(corrected_text) <= 9: # Türk plakaları için genel uzunluk
        return corrected_text
    
    # print(f"Post-processed OCR sonucu filtrelerden geçemedi: '{corrected_text}'")
    return ""


def read_plate_text(plate_image_roi: np.ndarray) -> str:
    if ocr_reader is None: return ""
    if plate_image_roi is None or plate_image_roi.size == 0: return ""
    try:
        preprocessed_plate = preprocess_plate_roi_for_ocr(plate_image_roi)
        if preprocessed_plate.size == 0: return ""

        # EasyOCR parametreleri ile oynamak önemli olabilir.
        results: List[str] = ocr_reader.readtext(
            preprocessed_plate, 
            detail=0, 
            paragraph=False,
            batch_size=8,           # Denenebilir (küçük resimler için 1 veya 4 daha iyi olabilir)
            workers=0,              # 0: ana thread, >0: paralel işleme (küçük resimler için gereksiz olabilir)
            allowlist=turkish_plate_chars, # Sadece bu karakterleri tanı
            # --- Denenebilecek diğer parametreler ---
            text_threshold=0.7,   # Metin algılama güven skoru (0-1)
            link_threshold=0.4,   # Bitişik karakterleri birleştirme skoru (0-1)
            contrast_ths=0.1,     # Kontrast eşiği (0-1), düşük değerler daha fazla metin bulur
            adjust_contrast=0.5,  # Kontrast ayarı (0-1)
            width_ths=0.5,        # Maksimum metin kutusu genişliği oranı (0-1)
            height_ths=0.5,       # Maksimum metin kutusu yüksekliği oranı (0-1)
            # x_ths=1.0,            # Bitişik metin kutuları için x-ekseni toleransı
            # y_ths=0.5,            # Bitişik metin kutuları için y-ekseni toleransı
            # mag_ratio=1.5         # Görüntü büyütme oranı
        )
        
        if results:
            raw_text = "".join(results).strip()
            # Ham metin üzerinde daha gelişmiş son işleme
            final_plate_text = postprocess_ocr_text(raw_text)
            return final_plate_text
        return ""
    except Exception as e:
        print(f"HATA: Plaka metni okunurken (OCR işlemi): {e}"); return ""

# ... (if __name__ == '__main__': bloğu önceki gibi kalabilir)

if __name__ == '__main__':
    print("OCR Module Test Başlatılıyor...")
    if not load_ocr_model():
        print("Test başarısız: OCR modeli yüklenemedi.")
        exit()

    test_plate_img = np.zeros((60, 200, 3), dtype=np.uint8)
    test_plate_img.fill(220)
    cv2.putText(test_plate_img, "34TRK001", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    print("\nTest plaka görüntüsü üzerinde metin okuma yapılıyor...")
    extracted_text = read_plate_text(test_plate_img)

    if extracted_text:
        print(f"Okunan Plaka Metni: '{extracted_text}' (Beklenen: 34TRK001)")
    else:
        print("Test plaka görüntüsünden metin okunamadı.")
    
    print("\nOCR Module Test Tamamlandı.")