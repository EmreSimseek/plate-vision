# PlateVision/config.py
import os

# Proje Ana Dizini (Bu dosyanın bulunduğu dizin)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Ayarları ---
# Kullanılacak YOLO modelinin adı (models/ klasöründe olmalı)
MODEL_FILENAME = 'v2_best.pt' # <<<--- KENDİ MODEL DOSYANIZIN ADINI BURAYA YAZIN
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_FILENAME)

DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Nesne tespiti için minimum güven eşiği (0.0 - 1.0)
PLATE_DETECTION_CONFIDENCE_THRESHOLD = 0.5 # Özellikle plaka tespiti için

# --- OCR Ayarları ---
OCR_LANGUAGES = ['en', 'tr'] # EasyOCR'ın kullanacağı diller

# --- Video Kaynağı ---
# data/ klasöründe olmalı veya tam yol verilmeli. '0' webcam için.
VIDEO_FILENAME = 'test_3dk.mp4' # <<<--- KULLANACAĞINIZ VİDEO DOSYASININ ADI
VIDEO_SOURCE = os.path.join(BASE_DIR, 'data', VIDEO_FILENAME)
# VIDEO_SOURCE = 0 # Canlı webcam için bu satırı aktif edin, üstteki satırı yorumlayın

# --- Veritabanı Ayarları ---
DATABASE_NAME = os.path.join(BASE_DIR, 'plate_vision.db')

# --- OCR Girdi Kayıt Ayarları ---
SAVE_OCR_INPUT_IMAGES = True  # True ise OCR'a giden plaka resimleri kaydedilir
OCR_INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'ocr_input_images') # Kayıt edilecek klasör

# --- Arayüz ve Uyarı Ayarları ---
# static/sounds/ klasöründe bu isimlerde .wav veya .mp3 dosyalarınız olmalı
SUCCESS_SOUND_FILENAME = 'success.wav' # veya success.mp3
ALARM_SOUND_FILENAME = 'alarm.wav'     # veya alarm.mp3

ALERT_SOUND_SUCCESS = os.path.join(BASE_DIR, 'static', 'sounds', SUCCESS_SOUND_FILENAME)
ALERT_SOUND_WARNING = os.path.join(BASE_DIR, 'static', 'sounds', ALARM_SOUND_FILENAME)

# --- ROI Ayarları ---
# ROI koordinatlarını saklamak için JSON dosyası
ROI_CONFIG_FILE = os.path.join(BASE_DIR, 'roi_config.json')


USE_CUSTOM_CLASS_MAP = True # VEYA False
# Eğer True ise, detection_module.py içindeki CUSTOM_CLASS_NAME_MAP'i doğru ayarlayın!

if __name__ == '__main__':
    print(f"Proje Ana Dizini: {BASE_DIR}")
    print(f"Model Yolu: {MODEL_PATH}")
    print(f"Video Kaynağı: {VIDEO_SOURCE}")
    print(f"Veritabanı Adı: {DATABASE_NAME}")
    print(f"Başarı Sesi Yolu: {ALERT_SOUND_SUCCESS}")
    print(f"Uyarı Sesi Yolu: {ALERT_SOUND_WARNING}")
    print(f"ROI Yapılandırma Dosyası: {ROI_CONFIG_FILE}")