# PlateVision/app.py
import cv2
import os
import time
import datetime
import json
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, flash
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from math import ceil
try:
    from playsound import playsound
except ImportError:
    playsound = None; print("UYARI: playsound kütüphanesi yok.")

import config
from core import detection_module, ocr_module, database_module 

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Global Değişkenler (Bir önceki mesajdaki gibi) ---
current_status_info: Dict[str, Any] = {
    "text_message": "Sistem başlatılıyor...", "message_class": "info",
    "detected_vehicle_type": None, "processed_plate_text": None,
    "ocr_detected_plate": None, "authorization_status": None,
    "last_truck_info_for_correction": None
}
models_loaded_successfully, db_initialized_successfully = False, False
last_played_sound_time, sound_cooldown = 0.0, 3
current_roi_points: List[List[int]] = []
video_frame_for_roi_selection: Optional[np.ndarray] = None
video_frame_width, video_frame_height = 640, 480
last_detected_boxes_for_drawing: List[Dict] = []
processed_track_ids: Dict[int, float] = {} 
TRACK_ID_PROCESS_TIMEOUT: int = 10 

# --- Yardımcı Fonksiyonlar (load_roi_from_file, save_roi_to_file, play_alert_sound_with_cooldown, initialize_components - Önceki gibi) ---
def load_roi_from_file(): # ... (önceki gibi)
    global current_roi_points
    if os.path.exists(config.ROI_CONFIG_FILE):
        try:
            with open(config.ROI_CONFIG_FILE, 'r') as f: data = json.load(f)
            if isinstance(data, list) and (len(data) == 0 or len(data) == 4):
                if len(data) == 4 and not all(isinstance(p,list) and len(p)==2 and all(isinstance(c,int) for c in p) for p in data):
                    current_roi_points = []; print(f"DEBUG: ROI dosya formatı (nokta içleri) geçersiz, ROI sıfırlandı: {data}")
                    return
                current_roi_points = data; # print(f"DEBUG: ROI yüklendi: {current_roi_points if current_roi_points else 'Tanımlı Değil'}")
            else: current_roi_points = []; print(f"DEBUG: ROI dosya formatı (ana yapı) geçersiz, ROI sıfırlandı: {data}")
        except Exception as e: print(f"HATA: ROI okuma: {e}"); current_roi_points = []
    else: current_roi_points = []; # print("DEBUG: ROI dosyası yok.")

def save_roi_to_file(roi_points: List[List[int]]) -> bool: # ... (önceki gibi)
    global current_roi_points
    try:
        with open(config.ROI_CONFIG_FILE, 'w') as f: json.dump(roi_points, f)
        current_roi_points = roi_points; print(f"DEBUG: ROI kaydedildi: {current_roi_points if current_roi_points else 'Sıfırlandı'}")
        return True
    except Exception as e: print(f"HATA: ROI kaydetme: {e}"); return False

def play_alert_sound_with_cooldown(sound_path: Optional[str]): # ... (önceki gibi)
    global last_played_sound_time; now = time.time()
    if playsound and sound_path and os.path.exists(sound_path) and (now-last_played_sound_time)>sound_cooldown:
        try: playsound(sound_path,block=False); last_played_sound_time=now
        except Exception: last_played_sound_time=now

def initialize_components(): # ... (önceki gibi)
    global models_loaded_successfully, db_initialized_successfully, current_status_info
    print("Bileşenler başlatılıyor...")
    models_loaded_successfully = detection_module.load_detection_model() and ocr_module.load_ocr_model()
    if not models_loaded_successfully: current_status_info.update({"text_message": "HATA: Modeller yüklenemedi.", "message_class": "danger"}); print("HATA: Modeller yüklenemedi.")
    db_initialized_successfully = database_module.init_db()
    if not db_initialized_successfully: current_status_info.update({"text_message": (current_status_info.get("text_message","") + "\nHATA: Veritabanı başlatılamadı.").strip(), "message_class": "danger"}); print("HATA: Veritabanı başlatılamadı.")
    load_roi_from_file()
    if models_loaded_successfully and db_initialized_successfully: current_status_info.update({"text_message": "Sistem hazır.", "message_class": "neutral"}); print("Tüm bileşenler başarıyla başlatıldı.")
    else: print("Bileşen başlatmada hatalar var.")
initialize_components()

def process_frame_logic(original_frame: np.ndarray):
    global current_status_info, current_roi_points, last_detected_boxes_for_drawing, processed_track_ids
    if not models_loaded_successfully or not db_initialized_successfully: return

    # --- DÜZELTİLMİŞ SATIR ---
    detections_list, _ = detection_module.detect_and_track_objects(original_frame.copy(), roi_coords=current_roi_points or None)
    last_detected_boxes_for_drawing = detections_list # Bu satır önemli, çizim için
    
    current_time = time.time()
    for tid in list(processed_track_ids.keys()): # Zaman aşımına uğramış ID'leri temizle
        if current_time - processed_track_ids[tid] > TRACK_ID_PROCESS_TIMEOUT:
            del processed_track_ids[tid]

    main_vehicle_to_process: Optional[Dict] = None
    for det in detections_list:
        label = det.get('label')
        track_id = det.get('track_id')
        if label == 'truck' and (track_id is None or track_id not in processed_track_ids):
            main_vehicle_to_process = det; break
    if not main_vehicle_to_process:
        for det in detections_list:
            label = det.get('label')
            track_id = det.get('track_id')
            if label == 'car' and (track_id is None or track_id not in processed_track_ids):
                main_vehicle_to_process = det; break
    
    frame_vehicle_type: Optional[str] = None; frame_ocr_plate: Optional[str] = None
    
    if main_vehicle_to_process:
        frame_vehicle_type = main_vehicle_to_process.get('label')
        vehicle_box = main_vehicle_to_process.get('box')
        for det_p in detections_list:
            if det_p.get('label')=='plate':
                p_box=det_p.get('box'); px_c,py_c=(p_box[0]+p_box[2])/2,(p_box[1]+p_box[3])/2
                if vehicle_box[0]<px_c<vehicle_box[2] and vehicle_box[1]<py_c<vehicle_box[3]:
                    y_p1,y_p2=max(0,p_box[1]),min(original_frame.shape[0],p_box[3]); x_p1,x_p2=max(0,p_box[0]),min(original_frame.shape[1],p_box[2])
                    if y_p1<y_p2 and x_p1<x_p2:
                        plate_image_roi=original_frame[y_p1:y_p2,x_p1:x_p2]
                        if plate_image_roi.size>0: frame_ocr_plate=ocr_module.read_plate_text(plate_image_roi)
                        if frame_ocr_plate: break 
    
    is_new_truck_event = (frame_vehicle_type == 'truck' and \
        (not current_status_info.get("last_truck_info_for_correction") or \
         current_status_info.get("last_truck_info_for_correction",{}).get("ocr_read_plate") != frame_ocr_plate or \
         current_status_info.get("last_truck_info_for_correction",{}).get("track_id") != (main_vehicle_to_process.get('track_id') if main_vehicle_to_process else None) or \
         (frame_ocr_plate is None and current_status_info.get("last_truck_info_for_correction",{}).get("ocr_read_plate") is not None) or \
         (frame_ocr_plate is not None and current_status_info.get("last_truck_info_for_correction",{}).get("ocr_read_plate") is None) \
        ) and (main_vehicle_to_process.get('track_id') not in processed_track_ids if main_vehicle_to_process and main_vehicle_to_process.get('track_id') is not None else True)
    )
    is_no_vehicle_or_car = not main_vehicle_to_process or frame_vehicle_type == 'car'

    if is_new_truck_event or is_no_vehicle_or_car:
        current_status_info["last_truck_info_for_correction"] = None

    if not main_vehicle_to_process: # İşlenecek yeni araç yoksa
        if not last_detected_boxes_for_drawing: # Ekranda da bir şey kalmadıysa
            current_status_info.update({"text_message":"Araç bekleniyor...","message_class":"neutral","detected_vehicle_type":None,"processed_plate_text":None,"ocr_detected_plate":None,"authorization_status":None,"last_truck_info_for_correction":None})
        return

    # Yeni araç için işlem devam ediyor
    current_status_info["detected_vehicle_type"] = frame_vehicle_type
    current_status_info["ocr_detected_plate"] = frame_ocr_plate
    current_status_info["processed_plate_text"] = frame_ocr_plate
    current_status_info["authorization_status"] = None
    
    new_text, new_class, sound, log_event = "Araç bekleniyor...", "neutral", None, None
    plate_for_db = frame_ocr_plate
    db_details_for_log: Optional[Dict] = None

    if frame_vehicle_type == 'truck':
        db_details_for_log_truck: Optional[Dict] = None
        if plate_for_db:
            auth, db_dets = database_module.check_plate_authorization(plate_for_db, 'truck')
            current_status_info["authorization_status"] = auth; db_details_for_log_truck = db_dets; db_details_for_log = db_dets
            new_text = f"KAMYON ({plate_for_db})\n"; log_event_base = "KAMYON_GIRIS"
            if auth=="AUTHORIZED": new_text+="YETKİLİ"; new_class="success"; sound=config.ALERT_SOUND_SUCCESS
            elif auth=="UNAUTHORIZED_DB": new_text+="YETKİSİZ (DB)"; new_class="danger"; sound=config.ALERT_SOUND_WARNING
            elif auth=="MISMATCHED_VEHICLE_TYPE": new_text+=f"DB'de '{db_dets.get('vehicle_type','?').upper() if db_dets else '?'}'!"; new_class="danger"; sound=config.ALERT_SOUND_WARNING
            elif auth=="NOT_REGISTERED": new_text+="KAYITLI DEĞİL"; new_class="warning"; sound=config.ALERT_SOUND_WARNING
            else: new_text+=f"DB Durumu: {auth}"; new_class="info"
            log_event = f"{log_event_base}_{auth}"
            current_status_info["last_truck_info_for_correction"] = {'ocr_read_plate':plate_for_db, 'track_id':main_vehicle_to_process.get('track_id'), 'vehicle_box':main_vehicle_to_process.get('box')}
        else:
            new_text = "KAMYON TESPİT\nPlaka Okunamadı"; new_class = "info"; log_event = "KAMYON_PLAKA_OKUNAMADI"
            current_status_info["last_truck_info_for_correction"] = {'ocr_read_plate':None, 'track_id':main_vehicle_to_process.get('track_id'), 'vehicle_box':main_vehicle_to_process.get('box')}
    elif frame_vehicle_type == 'car':
        current_status_info["authorization_status"] = "CAR_UNAUTHORIZED_POLICY"
        p_info = f" (Plaka: {frame_ocr_plate})" if frame_ocr_plate else ""
        new_text = f"OTOMOBİL{p_info}\nUYARI! YETKİSİZ."; new_class = "danger"; sound = config.ALERT_SOUND_WARNING
        log_event = "OTOMOBIL_YETKISIZ"
    elif main_vehicle_to_process:
        label = main_vehicle_to_process.get('label','BİLİNMEYEN')
        new_text = f"{label.upper()} TESPİT EDİLDİ."; new_class = "info"; log_event = f"{label.upper()}_TESPİT"
    
    current_status_info["text_message"] = new_text; current_status_info["message_class"] = new_class
    if sound: play_alert_sound_with_cooldown(sound)

    if log_event and main_vehicle_to_process:
        log_img_p = None
        try:
            v_b=main_vehicle_to_process.get('box');h,w=original_frame.shape[:2];x1,y1,x2,y2=v_b;x1,y1=max(0,x1),max(0,y1);x2,y2=min(w-1,x2),min(h-1,y2)
            if x1<x2 and y1<y2: r_img=original_frame[y1:y2,x1:x2];
            if r_img.size>0: ts=datetime.datetime.now().strftime("%y%m%d_%H%M%S%f");fn=f"{log_event.lower()}_{ts}.jpg";rp=os.path.join('static','log_images',fn);fp=os.path.join(config.BASE_DIR,rp);cv2.imwrite(fp,r_img);log_img_p=rp
        except Exception as e_li: print(f"Log resmi hatası: {e_li}")
        log_d_c = {"conf":main_vehicle_to_process.get('confidence'),"box":main_vehicle_to_process.get('box'), "track_id": main_vehicle_to_process.get('track_id')}
        if db_details_for_log: log_d_c["db_info"]=dict(db_details_for_log) # db_details_for_log_truck yerine genel
        database_module.log_activity(log_event, plate_for_db, frame_vehicle_type, log_img_p, json.dumps(log_d_c,default=str))
        
        track_id_processed = main_vehicle_to_process.get('track_id')
        if track_id_processed is not None:
            processed_track_ids[track_id_processed] = current_time


# --- generate_frames ve diğer Rotalar (Önceki "Tam Hal" mesajındaki gibi kalacak) ---
# generate_frames içindeki video döndürme ve çizim mantığı da aynı.
# Sadece emin olmak için generate_frames'in başını ve sonunu kontrol edin:
def generate_frames():
    global current_roi_points, video_frame_for_roi_selection, video_frame_width, video_frame_height, last_detected_boxes_for_drawing
    # ... (başlangıç kontrolleri ve cap açma) ...
    if not models_loaded_successfully or not db_initialized_successfully: # Hata frame'i
        err_f=np.zeros((480,640,3),np.uint8);cv2.putText(err_f,"Sistem Hatalı",(50,240),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        while True:_,b=cv2.imencode('.jpg',err_f);yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n');time.sleep(1)
    
    cap_s=config.VIDEO_SOURCE; cap_s=int(cap_s) if isinstance(cap_s,str) and cap_s.isdigit() else cap_s
    cap=cv2.VideoCapture(cap_s)
    if not cap.isOpened(): # Kaynak hatası
        err_f=np.zeros((480,640,3),np.uint8);cv2.putText(err_f,f"Kaynak Hatalı:{config.VIDEO_SOURCE}",(10,240),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        while True:_,b=cv2.imencode('.jpg',err_f);yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+b.tobytes()+b'\r\n');time.sleep(1)

    f_count, l_proc_int, roi_f_upd_int = 0, 5, 15
    first_frame_captured_for_dims = False

    while True:
        ret, frame_original = cap.read()
        if not ret:
            if isinstance(cap_s,str): cap.set(cv2.CAP_PROP_POS_FRAMES,0); ret,frame_original=cap.read()
            if not ret: print("Video akış sonu."); break
        
        frame = frame_original.copy()
        try:
            # --- VİDEO DÖNDÜRME: Sağa -90 derece (Saat Yönünde 90 Derece) ---
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) 
            # Eğer videonuz zaten doğru yöndeyse veya farklı bir döndürme/çevirme gerekiyorsa burayı ayarlayın.
            # Diğer seçenekler:
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Sola -90 (Saat Yönünün Tersi 90)
            # frame = cv2.rotate(frame, cv2.ROTATE_180) # 180 derece
            # frame = cv2.flip(frame, 1) # Yatay çevirme (ayna)
            # frame = cv2.flip(frame, 0) # Dikey çevirme
            pass 
        except Exception as e_rotate: print(f"Video transformasyon hatası: {e_rotate}"); frame = frame_original.copy()

        if not first_frame_captured_for_dims: # Boyutları DÖNDÜRÜLMÜŞ frame'den al
            video_frame_height, video_frame_width = frame.shape[:2]
            first_frame_captured_for_dims = True
            print(f"İşlenecek video çözünürlüğü (transformasyon sonrası): {video_frame_width}x{video_frame_height}")
        
        if f_count % roi_f_upd_int == 0: video_frame_for_roi_selection = frame.copy()
        
        if f_count % l_proc_int == 0: 
            process_frame_logic(frame.copy()) # Bu, last_detected_boxes_for_drawing'i günceller
            
        frame_to_display = frame.copy()
        if current_roi_points and len(current_roi_points) == 4:
            try: cv2.polylines(frame_to_display,[np.array([[int(p[0]),int(p[1])] for p in current_roi_points],np.int32)],True,(0,255,255),2)
            except: pass # ROI çizim hatası olursa görmezden gel
        
        # last_detected_boxes_for_drawing içindeki label'lar detection_module'den geliyor
        # ve orada CUSTOM_CLASS_MAP'e göre ayarlanmış olmalı.
        for det in last_detected_boxes_for_drawing: 
            label, conf, box, track_id = det.get('label'), det.get('confidence'), det.get('box'), det.get('track_id')
            if label and conf and box:
                x1,y1,x2,y2 = box; cmap={'plate':(255,165,0),'truck':(200,0,0),'car':(0,180,0)}; clr=cmap.get(label,(100,100,100)) # label (küçük harf) ile eşleşir
                cv2.rectangle(frame_to_display,(x1,y1),(x2,y2),clr,2)
                lbl_txt=f"{label}{f' ID:{track_id}' if track_id is not None else ''} {conf:.2f}"
                (w_t,h_t),_=cv2.getTextSize(lbl_txt,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
                txt_y=y1-10 if y1>h_t+12 else y1+h_t+12
                cv2.rectangle(frame_to_display,(x1,txt_y-h_t-2),(x1+w_t,txt_y+2),clr,-1)
                cv2.putText(frame_to_display,lbl_txt,(x1,txt_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        
        ret_encode,buffer_encode=cv2.imencode('.jpg',frame_to_display) # Değişken isimlerini değiştirdim
        if not ret_encode: 
            # print("[APP generate_frames] Frame encode edilemedi.") # DEBUG
            continue
        yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buffer_encode.tobytes()+b'\r\n'); f_count+=1
    cap.release()
    print("[APP] Video akışı generate_frames içinde sonlandı.")


# --- Diğer Rotalar (index, manage_plates, activity_logs, settings, get_roi_frame, set_roi, correct_plate_action) ---
# --- ve context_processor ile if __name__ == '__main__': ---
# BU KISIMLAR BİR ÖNCEKİ "TAM HAL" MESAJINDAKİ GİBİ KALACAK.
# Tekrar yazmıyorum, çok uzayacak. Sadece process_frame_logic ve generate_frames güncellendi.
# Emin olmak için index rotasını tekrar veriyorum:
@app.route('/')
def index():
    global current_roi_points
    return render_template('index.html', current_roi_for_logic=current_roi_points or [])

@app.route('/video_feed_route')
def video_feed_route(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status_route(): return jsonify(current_status_info)

@app.route('/manage_plates', methods=['GET', 'POST'])
def manage_plates_route():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add_plate':
            p_num, v_type, is_auth = request.form.get('plate_number'), request.form.get('vehicle_type'), (request.form.get('is_authorized') == 'yes')
            if p_num and v_type:
                if database_module.add_plate(p_num, v_type, is_auth): flash(f"'{p_num.upper()}' eklendi.", 'success')
                else: flash(f"'{p_num.upper()}' eklenemedi/mevcut.", 'warning')
            else: flash("Plaka ve Araç Türü zorunludur.", 'danger')
        elif action == 'delete_plate':
            p_id = request.form.get('plate_id_to_delete')
            if p_id and database_module.delete_plate(int(p_id)): flash("Plaka silindi.", 'success')
            else: flash("Plaka silinemedi.", 'danger')
        return redirect(url_for('manage_plates_route'))
    search_q, v_filter = request.args.get('search'), request.args.get('vehicle_type')
    plates = database_module.get_all_plates(search_term=search_q, vehicle_filter=v_filter)
    return render_template('manage_plates.html', plates=plates, search_query=search_q, vehicle_type_filter=v_filter)

@app.route('/activity_logs')
def activity_logs_route():
    page = request.args.get('page', 1, type=int)
    per_page = 20 # Sayfa başına log sayısı
    
    # Filtre parametrelerini al
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    event_type = request.args.get('event_type')
    plate = request.args.get('plate')
    vehicle_type = request.args.get('vehicle_type')

    logs, total_logs = database_module.get_activity_logs(
        limit=per_page, page=page,
        start_date=start_date, end_date=end_date,
        event_type_filter=event_type, plate_filter=plate,
        vehicle_type_filter=vehicle_type
    )
    
    total_pages = ceil(total_logs / per_page)

    # Formda kullanılacak olay türlerini dinamik olarak alabiliriz (opsiyonel)
    # event_types_distinct = ["KAMYON_GIRIS_AUTHORIZED", "OTOMOBIL_YETKISIZ", ...] 
    # Veya sabit bir liste olabilir.

    return render_template('activity_logs.html', 
                           logs=logs, 
                           current_page=page, 
                           total_pages=total_pages,
                           total_logs=total_logs,
                           # Filtre değerlerini şablona geri gönder (formda seçili kalmaları için)
                           start_date=start_date, end_date=end_date,
                           event_type=event_type, plate=plate, vehicle_type=vehicle_type
                           )

# YENİ ROTA: Tek bir log detayını göstermek için
@app.route('/log_detail/<int:log_id>')
def log_detail_route(log_id):
    log_entry = database_module.get_activity_log_by_id(log_id)
    if not log_entry:
        flash(f"ID {log_id} olan log kaydı bulunamadı.", "warning")
        return redirect(url_for('activity_logs_route'))
    return render_template('log_detail.html', log_entry=log_entry)

@app.route('/settings', methods=['GET'])
def settings_page_route():
    global video_frame_width, video_frame_height, current_roi_points
    return render_template('settings.html', frame_width=video_frame_width, frame_height=video_frame_height, current_roi=json.dumps(current_roi_points or []))

@app.route('/get_roi_frame')
def get_roi_frame_route():
    if video_frame_for_roi_selection is not None:
        ret, buf = cv2.imencode('.jpg', video_frame_for_roi_selection); 
        if ret: return Response(buf.tobytes(), mimetype='image/jpeg')
    return Response(status=204)

@app.route('/set_roi', methods=['POST'])
def set_roi_route():
    data = request.get_json(); points_req = data.get('roi_points')
    if points_req is not None and isinstance(points_req, list):
        if len(points_req) == 0:
            if save_roi_to_file([]): return jsonify(success=True, message="ROI sıfırlandı.", roi=[])
            return jsonify(success=False, message="ROI sıfırlanamadı.")
        elif len(points_req) == 4:
            try:
                valid_points = [[int(p[0]), int(p[1])] for p in points_req]
                if save_roi_to_file(valid_points): return jsonify(success=True, message="ROI ayarlandı.", roi=valid_points)
                return jsonify(success=False, message="ROI kaydedilemedi.")
            except: return jsonify(success=False, message="ROI veri formatı hatası.")
    return jsonify(success=False, message="Geçersiz ROI verisi.")

@app.route('/correct_plate_action', methods=['POST'])
def correct_plate_action_route():
    global current_status_info, processed_track_ids
    corrected_plate = request.form.get('corrected_plate_number', '').strip().upper()
    original_ocr_from_form = request.form.get('original_ocr_plate', None)
    track_id_from_form_str = request.form.get('track_id_for_correction', None) 
    track_id_corrected: Optional[int] = None
    if track_id_from_form_str and track_id_from_form_str.isdigit(): track_id_corrected = int(track_id_from_form_str)

    if not corrected_plate: flash("Düzeltilmiş plaka boş olamaz.", "warning"); return redirect(url_for('index'))
    last_truck_info = current_status_info.get("last_truck_info_for_correction")
    if not last_truck_info or \
       (last_truck_info.get('track_id') is not None and track_id_corrected is not None and last_truck_info.get('track_id') != track_id_corrected) or \
       current_status_info.get("detected_vehicle_type") != 'truck':
        flash("Düzeltme için aktif kamyon bilgisi eşleşmiyor/bulunamadı.", "info"); return redirect(url_for('index'))

    original_plate_for_log = last_truck_info.get('ocr_read_plate', original_ocr_from_form)
    if original_plate_for_log != corrected_plate and original_plate_for_log is not None : # Sadece gerçekten farklıysa ve orijinal varsa
        log_event_corr = "KAMYON_PLAKA_DUZELTME"; dets_corr = {"original_ocr":original_plate_for_log, "corrected_to":corrected_plate, "track_id":track_id_corrected, "by":"Görevli"}
        database_module.log_activity(log_event_corr, corrected_plate, 'truck', details=json.dumps(dets_corr))
        flash(f"Plaka '{original_plate_for_log}' -> '{corrected_plate}' olarak düzeltildi.", "success")
    elif not original_plate_for_log and corrected_plate: flash(f"Plaka '{corrected_plate}' olarak manuel girildi.", "info")
    
    current_status_info["processed_plate_text"] = corrected_plate
    auth_status, db_details = database_module.check_plate_authorization(corrected_plate, 'truck')
    current_status_info["authorization_status"] = auth_status
    new_text = f"KAMYON ({corrected_plate}) - DÜZELTME SONRASI\n"; new_class="info"; sound=None
    log_event_after_corr = f"KAMYON_DUZELTILMIS_GIRIS_{auth_status}" 
    if auth_status=="AUTHORIZED": new_text+="YETKİLİ"; new_class="success"; sound=config.ALERT_SOUND_SUCCESS # ... (diğer durumlar)
    elif auth_status == "UNAUTHORIZED_DB": new_text+="YETKİSİZ (DB)"; new_class="danger"; sound=config.ALERT_SOUND_WARNING
    elif auth_status == "NOT_REGISTERED": new_text+="KAYITLI DEĞİL"; new_class="warning"; sound=config.ALERT_SOUND_WARNING
    elif auth_status == "MISMATCHED_VEHICLE_TYPE" and db_details: new_text+=f"DB'de '{db_details.get('vehicle_type','?').upper()}'!"; new_class="danger";sound=config.ALERT_SOUND_WARNING
    else: new_text+=f"DB Sonucu: {auth_status}"; new_class="warning"; sound=config.ALERT_SOUND_WARNING
    
    current_status_info["text_message"] = new_text; current_status_info["message_class"] = new_class
    if sound: play_alert_sound_with_cooldown(sound)
    
    log_img_p_corr: Optional[str]=None; v_box_log=last_truck_info.get("vehicle_box"); frame_log=video_frame_for_roi_selection
    if v_box_log and frame_log is not None:
        try:
            h,w=frame_log.shape[:2];x1,y1,x2,y2=v_box_log;x1,y1=max(0,x1),max(0,y1);x2,y2=min(w-1,x2),min(h-1,y2)
            if x1<x2 and y1<y2: r_img=frame_log[y1:y2,x1:x2];
            if r_img.size>0: ts=datetime.datetime.now().strftime("%y%m%d_%H%M%S%f");fn=f"{log_event_after_corr.lower()}_{ts}.jpg";rp=os.path.join('static','log_images',fn);fp=os.path.join(config.BASE_DIR,rp);cv2.imwrite(fp,r_img);log_img_p_corr=rp
        except Exception as e_lic: print(f"Düzeltme log resmi hatası: {e_lic}")
    log_dets_main={"corrected_plate":corrected_plate, "track_id":track_id_corrected}; 
    if original_plate_for_log != corrected_plate: log_dets_main["original_ocr_if_diff"]=original_plate_for_log
    if db_details: log_dets_main["db_info"]=dict(db_details)
    database_module.log_activity(log_event_after_corr,corrected_plate,'truck',log_img_p_corr,json.dumps(log_dets_main,default=str))
    
    if track_id_corrected is not None: processed_track_ids[track_id_corrected] = time.time()
    current_status_info["last_truck_info_for_correction"] = None 
    return redirect(url_for('index'))

@app.context_processor
def inject_global_vars_for_templates(): return dict(now=datetime.datetime.utcnow())

if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False, host='0.0.0.0', port=5000)