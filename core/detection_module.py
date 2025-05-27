# PlateVision/core/detection_module.py
import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# --- Python PATH Ayarı (Proje kök dizinini sys.path'e ekler) ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- END Python PATH Ayarı ---
import config # config.py'yi import et

yolo_model: Optional[object] = None # YOLO model objesi için tip

# !!! ÖNEMLİ: EĞER config.USE_CUSTOM_CLASS_MAP = True İSE BU MAP KULLANILIR !!!
# Modelinizin döndürdüğü GERÇEK SINIF ID'LERİNİ, sizin istediğiniz STRING ETİKETLERE eşleyin.
# Örnek (Test kodunuzdaki `fix_map = {0: 'car', 1: 'plate', 2: 'truck'}` ise ve modeliniz de bu ID'leri veriyorsa):
CUSTOM_CLASS_NAME_MAP: Dict[int, str] = {
    0: 'car',    # Modelden gelen ID 0'ı 'car' olarak etiketle
    1: 'plate',  # Modelden gelen ID 1'i 'plate' olarak etiketle
    2: 'truck'   # Modelden gelen ID 2'yi 'truck' olarak etiketle
    # Eğer modelinizin ID -> İsim eşlemesi farklıysa veya daha fazla sınıfınız varsa burayı DÜZENLEYİN!
}
# Örneğin, eğer modelinizin `names` özelliği {0: 'KAMYON', 1: 'ARABA', 2: 'PLAKA'} ise
# ve siz bunları küçük harf ve İngilizce istiyorsanız, map şöyle olabilir:
# CUSTOM_CLASS_NAME_MAP = {0: 'truck', 1: 'car', 2: 'plate'}

def load_detection_model() -> bool:
    """YOLO modelini yükler ve hangi cihazda çalıştığını loglar."""
    global yolo_model
    model_path = config.MODEL_PATH
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        yolo_model = None
        return False
    
    try:
        from ultralytics import YOLO
        import torch

        # Cihazı belirle
        device_to_try = 'cpu'
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"PyTorch CUDA'yı başarıyla buldu. Kullanılabilir GPU: {gpu_name}")
                device_to_try = 'cuda' # veya spesifik GPU için 'cuda:0'
            except Exception as e_gpu:
                print(f"CUDA mevcut görünüyor ama GPU adı alınamadı: {e_gpu}. CPU'ya dönülüyor.")
        else:
            print("PyTorch CUDA'yı bulamadı veya CUDA destekli GPU yok. CPU kullanılacak.")

        # Modeli yükle. Ultralytics genellikle device'ı otomatik algılar.
        # `device` parametresini YOLO() constructor'ına vermek yerine,
        # model yüklendikten sonra `.to(device)` ile taşımak daha esnek olabilir,
        # ama çoğu zaman YOLO() çağrısı yeterlidir.
        print(f"YOLO modeli yükleniyor: {model_path} (Hedef cihaz: {device_to_try})")
        yolo_model = YOLO(model_path)
        # yolo_model.to(device_to_try) # Gerekirse modeli explicit olarak cihaza taşıyın.
                                     # Çoğu durumda YOLO() bunu zaten yapar.

        # Modelin gerçekten hangi cihazda olduğunu kontrol et
        actual_device_type = "Bilinmiyor"
        if hasattr(yolo_model, 'device') and yolo_model.device is not None:
            if hasattr(yolo_model.device, 'type'): # torch.device objesi
                actual_device_type = str(yolo_model.device.type)
            elif isinstance(yolo_model.device, str): # string ise
                actual_device_type = yolo_model.device
        
        if 'cuda' in actual_device_type.lower() and torch.cuda.is_available():
            print(f"YOLO Modeli başarıyla GPU üzerinde çalışacak şekilde ayarlandı.")
        elif 'cpu' in actual_device_type.lower():
            print(f"YOLO Modeli GPU üzerinde çalışacak.")
        else: # Eğer device özelliği belirsizse ama CUDA varsa, GPU'da olabilir
            if torch.cuda.is_available() and device_to_try == 'cuda':
                print(f"YOLO Modelinin cihazı tam belirlenemedi ama CUDA etkin olduğu için GPU'da çalışması bekleniyor.")
            else:
                print(f"YOLO Modelinin cihazı tam belirlenemedi, CPU üzerinde çalışması bekleniyor.")


        # Sınıf isimlerini kontrol et
        model_names_from_yolo = getattr(yolo_model, 'names', None)
        if model_names_from_yolo and isinstance(model_names_from_yolo, dict) and model_names_from_yolo:
            print(f"Modelin kendi sınıf isimleri (model.names): {model_names_from_yolo}")
        else:
            print(f"UYARI: Modelin kendi sınıf isimleri (model.names) bulunamadı veya geçersiz.")
        
        use_custom_map = getattr(config, 'USE_CUSTOM_CLASS_MAP', False)
        if use_custom_map:
            print(f"config.USE_CUSTOM_CLASS_MAP=True olduğu için ÖZEL SINIF EŞLEMESİ kullanılacak: {CUSTOM_CLASS_NAME_MAP}")
        else:
            print("config.USE_CUSTOM_CLASS_MAP=False olduğu için model.names (varsa) kullanılacak.")
        
        return True

    except ImportError:
        print("HATA: 'ultralytics' veya 'torch' kütüphanesi kurulu değil.")
        yolo_model = None
        return False
    except Exception as e:
        print(f"HATA: Model yüklenirken genel bir sorun oluştu ({model_path}): {e}")
        yolo_model = None
        return False

def get_class_name(class_id: int, model_names_dict: Optional[Dict] = None) -> str:
    """
    Verilen class_id için sınıf adını belirler.
    config.USE_CUSTOM_CLASS_MAP True ise CUSTOM_CLASS_NAME_MAP'i öncelikli kullanır.
    Değilse veya ID map'te yoksa model_names_dict'i kullanır.
    Hiçbiri yoksa ID'yi string olarak döndürür.
    """
    use_custom_map = getattr(config, 'USE_CUSTOM_CLASS_MAP', False)

    if use_custom_map:
        if class_id in CUSTOM_CLASS_NAME_MAP:
            return CUSTOM_CLASS_NAME_MAP[class_id]
        else:
            # Özel map kullanılmak isteniyor ama ID map'te yok.
            # Bu durumda modelin kendi ismine bakmak yerine doğrudan "Bilinmeyen" demek daha doğru olabilir
            # çünkü kullanıcı özel bir maplama istediğini belirtmiş.
            print(f"UYARI: ID {class_id} özel CUSTOM_CLASS_NAME_MAP'te bulunamadı!")
            # Alternatif olarak model.names'e bakılabilir:
            # if model_names_dict and class_id in model_names_dict:
            #     return model_names_dict[class_id]
            return f"ID_MAP_DISI:{class_id}" 

    # Özel map kullanılmıyorsa veya ID özel map'te bulunamadıysa (yukarıdaki mantığa göre buraya düşmez eğer özel map aktifse)
    if model_names_dict and isinstance(model_names_dict, dict) and class_id in model_names_dict:
        return model_names_dict[class_id]
        
    return f"ID_TANIMSIZ:{class_id}" # Hiçbir yerden bulunamazsa

def detect_and_track_objects(frame: np.ndarray, roi_coords: Optional[List[List[int]]] = None) -> Tuple[List[Dict], np.ndarray]:
    """
    Verilen video karesi üzerinde nesne tespiti ve takibi yapar.
    ROI tanımlıysa sadece o bölgede çalışır.
    Tespit edilen nesnelerin etiketlerini, güven skorlarını, kutularını ve takip ID'lerini döndürür.
    Ayrıca, bu bilgilerin çizildiği bir frame kopyası da döndürür.
    """
    if yolo_model is None:
        # print("[DETECTION WARNING] YOLO modeli yüklenmemiş, boş sonuç dönülüyor.")
        return [], frame.copy()

    processed_frame = frame.copy() # Üzerine çizim yapılacak kopya
    frame_for_detection = frame.copy() # Tespit için kullanılacak kare (ROI uygulanabilir)
    
    # ROI Uygulama
    if roi_coords and len(roi_coords) == 4:
        try:
            # Gelen koordinatların int olduğundan emin ol (JS'den float gelebilir)
            roi_poly_np = np.array([[int(p[0]), int(p[1])] for p in roi_coords], dtype=np.int32)
            cv2.polylines(processed_frame, [roi_poly_np], True, (0, 255, 255), 2) # ROI'yi ana frame'e çiz
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [roi_poly_np], (255))
            frame_for_detection = cv2.bitwise_and(frame, frame, mask=mask) # Tespit için maskelenmiş frame
        except Exception as e: 
            print(f"ROI uygulama hatası (tüm frame kullanılacak): {e}")
            frame_for_detection = frame.copy()

    detections_output: List[Dict] = []
    try:
        # YOLOv8 ile Takip (tracker belirtilmezse varsayılanını kullanır, örn: botsort.yaml)
        results_from_model = yolo_model.track(source=frame_for_detection, persist=True, verbose=False) 
        
        if results_from_model and len(results_from_model) > 0:
            processed_results = results_from_model[0] # Genelde tek resim için tek sonuç
            model_internal_names = getattr(yolo_model, 'names', None) # Modelin kendi sınıf isimleri
            
            object_ids_tensor = None
            if hasattr(processed_results.boxes, 'is_track') and processed_results.boxes.is_track:
                object_ids_tensor = processed_results.boxes.id # Bu bir Tensor olabilir veya None

            # Kutuları, sınıfları, skorları ve ID'leri al
            boxes_data = processed_results.boxes.data.cpu().numpy() # Tüm veriyi (xyxy, id, conf, cls) içerir

            for i in range(len(boxes_data)):
                box_info = boxes_data[i]
                x1, y1, x2, y2 = map(int, box_info[:4])
                
                track_id: Optional[int] = None
                # Ultralytics'in farklı versiyonlarında 'id' sütununun indeksi değişebilir.
                # Genellikle .data içinde 4. veya 5. index (0-based) olabilir.
                # Veya object_ids_tensor kullanılıyorsa:
                if object_ids_tensor is not None and i < len(object_ids_tensor):
                    track_id = int(object_ids_tensor[i])
                elif len(box_info) >= 5 and boxes_data.is_track: # Eğer .data içinde ID varsa (eski versiyonlar)
                    # Bu kısım versiyona göre ayarlanmalı, en iyisi object_ids_tensor'a güvenmek.
                    # Şimdilik, eğer object_ids_tensor yoksa track_id None kalacak.
                    pass

                score = float(box_info[-2]) # Genelde sondan ikinci confidence olur
                class_id = int(box_info[-1]) # Genelde sonuncusu class id olur
                
                # Sınıf adını belirle
                class_name_from_logic = get_class_name(class_id, model_internal_names)
                final_display_label = class_name_from_logic.lower() # Sistemde küçük harf kullan

                current_conf_thresh = config.PLATE_DETECTION_CONFIDENCE_THRESHOLD if final_display_label == 'plate' else config.DETECTION_CONFIDENCE_THRESHOLD
                if score >= current_conf_thresh:
                    detections_output.append({
                        'label': final_display_label, 
                        'confidence': score, 
                        'box': [x1, y1, x2, y2], 
                        'track_id': track_id
                    })
                    
                    # Çizim `processed_frame` üzerine (ROI çizgili olan ana frame)
                    cmap={'plate':(255,165,0),'truck':(200,0,0),'car':(0,180,0)}; color=cmap.get(final_display_label,(100,100,100))
                    cv2.rectangle(processed_frame,(x1,y1),(x2,y2),color,2)
                    
                    label_text_on_box = f"{final_display_label}"
                    if track_id is not None: label_text_on_box += f" ID:{track_id}"
                    label_text_on_box += f" {score:.2f}"
                    
                    (w_text,h_text), baseline = cv2.getTextSize(label_text_on_box, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_y_pos = y1 - 10 if y1 > (h_text + 10) else y1 + h_text + baseline + 5 # Etiket pozisyonu
                    
                    # Etiket için arka plan
                    cv2.rectangle(processed_frame, (x1, text_y_pos - h_text - baseline -2), (x1 + w_text, text_y_pos + baseline -2), color, -1)
                    cv2.putText(processed_frame, label_text_on_box, (x1, text_y_pos - baseline//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) # Beyaz yazı
    except Exception as e: 
        print(f"HATA: Nesne takip/tespiti/çizimi sırasında: {e}")
        import traceback
        traceback.print_exc() # Tam hata izini yazdır
        return [], frame.copy() # Hata durumunda orijinal frame ve boş liste
        
    return detections_output, processed_frame