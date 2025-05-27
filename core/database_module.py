# PlateVision/core/database_module.py
import sqlite3
import os
import sys
from typing import Union, List, Dict, Optional, Tuple
import datetime # datetime modülünü import et
import json
# --- Python PATH Ayarı (Proje kök dizinini sys.path'e ekler) ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- END Python PATH Ayarı ---
import config # config.py'yi import et

def get_db_connection() -> sqlite3.Connection:
    """Veritabanına bir bağlantı oluşturur ve döndürür."""
    conn = sqlite3.connect(config.DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Sütun isimleriyle erişim için
    return conn

def init_db() -> bool:
    """
    Veritabanını ve gerekli tabloları (eğer yoksa) oluşturur/günceller.
    Uygulama başladığında bir kez çağrılması yeterlidir.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Plates tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE NOT NULL,
                vehicle_type TEXT NOT NULL CHECK(vehicle_type IN ('truck', 'car')),
                is_authorized INTEGER NOT NULL DEFAULT 0, -- 0: Yetkisiz, 1: Yetkili
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Mevcut 'plates' tablosunu kontrol et ve gerekirse sütunları ekle
        table_info = cursor.execute("PRAGMA table_info(plates)").fetchall()
        column_names = [info['name'] for info in table_info]
        if 'vehicle_type' not in column_names:
            cursor.execute("ALTER TABLE plates ADD COLUMN vehicle_type TEXT NOT NULL DEFAULT 'truck' CHECK(vehicle_type IN ('truck', 'car'))")
            print("'vehicle_type' sütunu 'plates' tablosuna eklendi.")
        if 'is_authorized' not in column_names:
            cursor.execute("ALTER TABLE plates ADD COLUMN is_authorized INTEGER NOT NULL DEFAULT 0")
            print("'is_authorized' sütunu 'plates' tablosuna eklendi.")

        # Activity Log tablosu
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                plate_number TEXT,
                vehicle_type_detected TEXT,
                image_path TEXT,
                details TEXT -- JSON formatında ekstra detaylar
            )
        ''')
        conn.commit()
        print(f"Veritabanı '{config.DATABASE_NAME}' başarıyla başlatıldı/kontrol edildi.")
        
        # Aktivite logları için resimlerin saklanacağı klasörü oluştur
        log_images_dir = os.path.join(config.BASE_DIR, 'static', 'log_images')
        os.makedirs(log_images_dir, exist_ok=True)
        return True
    except sqlite3.Error as e:
        print(f"HATA: Veritabanı başlatılırken/oluşturulurken: {e}")
        return False
    finally:
        if conn:
            conn.close()

def add_plate(plate_number: str, vehicle_type: str, is_authorized: bool) -> bool:
    """Verilen plaka numarasını, araç türünü ve yetki durumunu veritabanına ekler."""
    if not all([plate_number, isinstance(plate_number, str), vehicle_type in ['truck', 'car']]):
        print("Uyarı: Geçersiz plaka, araç türü veya parametreler.")
        return False
    normalized_plate = ''.join(filter(str.isalnum, plate_number)).upper()
    if not normalized_plate:
        print("Uyarı: Normalleştirilmiş plaka numarası boş.")
        return False
    auth_value = 1 if is_authorized else 0
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO plates (plate_number, vehicle_type, is_authorized) VALUES (?, ?, ?)",
            (normalized_plate, vehicle_type, auth_value)
        )
        conn.commit()
        print(f"Plaka '{normalized_plate}' ({vehicle_type}, Yetkili: {is_authorized}) başarıyla eklendi.")
        return True
    except sqlite3.IntegrityError: # Plaka zaten var (UNIQUE kısıtlaması)
        print(f"Bilgi: Plaka '{normalized_plate}' zaten veritabanında mevcut.")
        return False 
    except sqlite3.Error as e:
        print(f"HATA: Plaka '{normalized_plate}' eklenirken bir sorun oluştu: {e}")
        return False
    finally:
        if conn:
            conn.close()

def _parse_datetime_str(datetime_str: Optional[str], field_name_for_log: str = "tarih/zaman") -> Optional[datetime.datetime]:
    """Yardımcı fonksiyon: Veritabanından gelen tarih string'ini datetime objesine çevirir."""
    if not isinstance(datetime_str, str) or not datetime_str.strip():
        # print(f"[DEBUG] _parse_datetime_str: Boş veya string olmayan değer alındı: '{datetime_str}' (alan: {field_name_for_log})")
        return None
    
    # SQLite'tan gelen yaygın formatlar
    formats_to_try = [
        '%Y-%m-%d %H:%M:%S.%f', # Milisaniyeli format
        '%Y-%m-%d %H:%M:%S'     # Milisaniyesiz format
    ]
    
    for fmt in formats_to_try:
        try:
            return datetime.datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue # Bir sonraki formatı dene
            
    # Eğer hiçbir format uymadıysa, uyarı ver ve None döndür
    print(f"UYARI: _parse_datetime_str: Desteklenmeyen {field_name_for_log} formatı - Alınan Değer: '{datetime_str}'. Parse edilemedi.")
    return None

def get_plate_details(plate_number: str) -> Optional[Dict]:
    """Verilen plaka numarasının tüm detaylarını (added_date datetime objesi olarak) döndürür."""
    if not plate_number or not isinstance(plate_number, str): return None
    normalized_plate = ''.join(filter(str.isalnum, plate_number)).upper()
    if not normalized_plate: return None
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, plate_number, vehicle_type, is_authorized, added_date FROM plates WHERE plate_number = ?",
            (normalized_plate,)
        )
        row = cursor.fetchone()
        if row:
            plate_data = dict(row)
            plate_data['added_date'] = _parse_datetime_str(plate_data.get('added_date'), "plate.added_date")
            return plate_data
        return None
    except sqlite3.Error as e:
        print(f"HATA: Plaka '{normalized_plate}' detayları sorgulanırken: {e}")
        return None
    finally:
        if conn:
            conn.close()

def check_plate_authorization(plate_number: str, expected_vehicle_type: str) -> Tuple[str, Optional[Dict]]:
    """Plakayı kontrol eder ve yetkilendirme durumunu ve plaka detaylarını döndürür."""
    details = get_plate_details(plate_number) # Bu fonksiyon zaten datetime objesi içeren 'added_date' döndürür
    if not details:
        return "NOT_REGISTERED", None

    db_vehicle_type = details.get('vehicle_type')
    db_is_authorized = bool(details.get('is_authorized'))

    if expected_vehicle_type == 'truck':
        if db_vehicle_type != 'truck':
            return "MISMATCHED_VEHICLE_TYPE", details
        if db_is_authorized:
            return "AUTHORIZED", details
        else:
            return "UNAUTHORIZED_DB", details
    
    # Diğer araç türleri için (örn: 'car') veya genel bir "bulundu" durumu
    # if expected_vehicle_type == 'car' and db_vehicle_type == 'car' and db_is_authorized:
    #    return "AUTHORIZED_CAR_DB", details # Eğer arabalar için de yetkilendirme istenirse

    return "INFO_DB_DETAILS_FOUND", details # Plaka DB'de var ama yukarıdaki koşullara uymadı

def get_all_plates(search_term: Optional[str] = None, vehicle_filter: Optional[str] = None) -> List[Dict]:
    """
    Veritabanındaki tüm plakaları veya filtrelenmiş plakaları (added_date datetime objesi olarak) döndürür.
    """
    plates_list: List[Dict] = []
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT id, plate_number, vehicle_type, is_authorized, added_date FROM plates"
        conditions: List[str] = []
        params: List[Union[str, int]] = []

        if search_term:
            conditions.append("plate_number LIKE ?")
            params.append(f"%{search_term.upper()}%")

        if vehicle_filter and vehicle_filter in ['truck', 'car']:
            conditions.append("vehicle_type = ?")
            params.append(vehicle_filter)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY added_date DESC"
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        for row_data in rows:
            plate_entry = dict(row_data)
            plate_entry['added_date'] = _parse_datetime_str(plate_entry.get('added_date'), "plate.added_date")
            plates_list.append(plate_entry)
        return plates_list
    except sqlite3.Error as e:
        print(f"HATA: Plakalar alınırken (filtreli): {e}")
        return []
    finally:
        if conn:
            conn.close()

def delete_plate(plate_id: int) -> bool:
    """Verilen ID'ye sahip plakayı veritabanından siler."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM plates WHERE id = ?", (plate_id,))
        conn.commit()
        return conn.changes() > 0 # Silinen satır sayısı 0'dan büyükse True
    except sqlite3.Error as e:
        print(f"HATA: ID {plate_id} olan plaka silinirken: {e}")
        return False
    finally:
        if conn:
            conn.close()

def log_activity(event_type: str, plate_number: Optional[str] = None, 
                 vehicle_type_detected: Optional[str] = None, 
                 image_path: Optional[str] = None, details: Optional[str] = None) -> bool:
    """Veritabanına bir aktivite kaydı ekler."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO activity_log (event_type, plate_number, vehicle_type_detected, image_path, details) VALUES (?, ?, ?, ?, ?)",
            (event_type, plate_number, vehicle_type_detected, image_path, details)
        )
        conn.commit()
        # print(f"[LOG] Event: {event_type}, Plate: {plate_number if plate_number else '-'}") # Detaylı log için
        return True
    except sqlite3.Error as e:
        print(f"HATA: Aktivite loglanırken: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_activity_logs(limit: int = 50, page: int = 1, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      event_type_filter: Optional[str] = None,
                      plate_filter: Optional[str] = None,
                      vehicle_type_filter: Optional[str] = None) -> Tuple[List[Dict], int]:
    """Filtrelenmiş ve sayfalanmış aktivite loglarını ve toplam sayısını döndürür."""
    logs_list: List[Dict] = []
    conn = None
    offset = (page - 1) * limit
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        base_query = "FROM activity_log"
        conditions: List[str] = []
        params: List[any] = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(f"{start_date} 00:00:00") # Günün başlangıcı
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(f"{end_date} 23:59:59") # Günün sonu
        if event_type_filter:
            conditions.append("event_type LIKE ?")
            params.append(f"%{event_type_filter}%")
        if plate_filter:
            conditions.append("plate_number LIKE ?")
            params.append(f"%{plate_filter.upper()}%")
        if vehicle_type_filter:
            conditions.append("vehicle_type_detected = ?")
            params.append(vehicle_type_filter.lower())

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        
        # Önce toplam sayıyı al (sayfalama için)
        count_query = f"SELECT COUNT(*) as total_count {base_query}{where_clause}"
        cursor.execute(count_query, tuple(params))
        total_count = cursor.fetchone()['total_count']

        # Sonra belirli sayfadaki logları al
        data_query = f"SELECT id, timestamp, event_type, plate_number, vehicle_type_detected, image_path, details {base_query}{where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params_with_limit = params + [limit, offset]
        
        cursor.execute(data_query, tuple(params_with_limit))
        rows = cursor.fetchall()
        
        for row_data in rows:
            log_entry = dict(row_data)
            log_entry['timestamp'] = _parse_datetime_str(log_entry.get('timestamp'), "log.timestamp")
            logs_list.append(log_entry)
            
        return logs_list, total_count
    except sqlite3.Error as e:
        print(f"HATA: Aktivite logları (filtreli) alınırken: {e}")
        return [], 0
    finally:
        if conn: conn.close()


def get_activity_log_by_id(log_id: int) -> Optional[Dict]:
    """Belirli bir ID'ye sahip aktivite logunu döndürür."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM activity_log WHERE id = ?", (log_id,))
        row = cursor.fetchone()
        if row:
            log_entry = dict(row)
            log_entry['timestamp'] = _parse_datetime_str(log_entry.get('timestamp'), "log.timestamp")
            # Detayları JSON ise parse et (opsiyonel)
            if log_entry.get('details'):
                try:
                    log_entry['details_parsed'] = json.loads(log_entry['details'])
                except json.JSONDecodeError:
                    log_entry['details_parsed'] = {"raw": log_entry['details']} # Parse edilemezse ham string
            return log_entry
        return None
    except sqlite3.Error as e:
        print(f"HATA: ID {log_id} olan log alınırken: {e}")
        return None
    finally:
        if conn: conn.close()      
from math import ceil # En üste import math

def get_activity_logs_paginated_filtered(
    page: int = 1, 
    per_page: int = 20, # config'den de alınabilir
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    event_type_filter: Optional[str] = None,
    plate_filter: Optional[str] = None,
    vehicle_type_filter: Optional[str] = None
) -> Tuple[List[Dict], int]:
    """Aktivite loglarını sayfalama ve filtreleme ile döndürür."""
    logs_list: List[Dict] = []
    conn = None
    offset = (page - 1) * per_page
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        base_query = "FROM activity_log"
        conditions: List[str] = []
        params: List[any] = [] # Union[str, int] yerine Any daha genel olabilir

        if start_date_str:
            try: 
                # Tarih formatını doğrula ve sorguya uygun hale getir
                start_dt_obj = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
                conditions.append("DATE(timestamp) >= ?") # Sadece tarih kısmını karşılaştır
                params.append(start_dt_obj.strftime('%Y-%m-%d'))
            except ValueError: print(f"Geçersiz başlangıç tarihi formatı: {start_date_str}")
        
        if end_date_str:
            try:
                end_dt_obj = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
                conditions.append("DATE(timestamp) <= ?")
                params.append(end_dt_obj.strftime('%Y-%m-%d'))
            except ValueError: print(f"Geçersiz bitiş tarihi formatı: {end_date_str}")

        if event_type_filter and event_type_filter.strip():
            conditions.append("LOWER(event_type) LIKE ?")
            params.append(f"%{event_type_filter.lower().strip()}%")
            
        if plate_filter and plate_filter.strip():
            conditions.append("UPPER(plate_number) LIKE ?")
            params.append(f"%{plate_filter.upper().strip()}%")

        if vehicle_type_filter and vehicle_type_filter.strip() in ['truck', 'car', 'unknown']:
            conditions.append("LOWER(vehicle_type_detected) = ?")
            params.append(vehicle_type_filter.lower().strip())

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        
        count_query = f"SELECT COUNT(*) as total_count {base_query}{where_clause}"
        cursor.execute(count_query, tuple(params))
        total_logs = 0
        total_logs_result = cursor.fetchone()
        if total_logs_result:
            total_logs = total_logs_result['total_count']

        data_query = f"SELECT id, timestamp, event_type, plate_number, vehicle_type_detected, image_path, details {base_query}{where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params_for_data = list(params) # Parametre listesinin kopyasını al
        params_for_data.append(per_page)
        params_for_data.append(offset)
        
        cursor.execute(data_query, tuple(params_for_data))
        rows = cursor.fetchall()
        
        for row_data in rows:
            log_entry = dict(row_data)
            log_entry['timestamp'] = _parse_datetime_str(log_entry.get('timestamp'), "log.timestamp")
            # Detayları JSON ise parse et (log_detail.html için)
            if log_entry.get('details'):
                try: log_entry['details_parsed'] = json.loads(log_entry['details'])
                except json.JSONDecodeError: log_entry['details_parsed'] = {"raw_details": log_entry['details']}
            logs_list.append(log_entry)
            
        return logs_list, total_logs
    except sqlite3.Error as e:
        print(f"HATA: Filtreli/Sayfalı logları alma: {e}")
        return [], 0
    finally:
        if conn: conn.close()

def get_distinct_event_types() -> List[str]:
    """Veritabanındaki benzersiz olay türlerini (boş olmayan) döndürür."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT event_type FROM activity_log WHERE event_type IS NOT NULL AND event_type != '' ORDER BY event_type")
        rows = cursor.fetchall()
        return [row['event_type'] for row in rows]
    except sqlite3.Error as e:
        print(f"HATA: Benzersiz olay türlerini alma: {e}")
        return []
    finally:
        if conn: conn.close()

def get_activity_log_by_id(log_id: int) -> Optional[Dict]:
    """Belirli bir ID'ye sahip aktivite logunu detaylarıyla döndürür."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM activity_log WHERE id = ?", (log_id,))
        row = cursor.fetchone()
        if row:
            log_entry = dict(row)
            log_entry['timestamp'] = _parse_datetime_str(log_entry.get('timestamp'), "log.timestamp")
            if log_entry.get('details'):
                try:
                    log_entry['details_parsed'] = json.loads(log_entry['details'])
                except json.JSONDecodeError:
                    log_entry['details_parsed'] = {"raw_details": log_entry['details']} 
            else:
                log_entry['details_parsed'] = {} # Detay yoksa boş dict
            return log_entry
        return None
    except sqlite3.Error as e:
        print(f"HATA: Log ID {log_id} alınırken: {e}")
        return None
    finally:
        if conn: conn.close()          

if __name__ == '__main__':
    # Test amaçlı veritabanını silip yeniden oluşturabilirsiniz
    # db_file = os.path.join(project_root_dir, config.DATABASE_NAME) # project_root_dir'i kullan
    # if os.path.exists(db_file):
    #     os.remove(db_file)
    #     print(f"Test için eski veritabanı '{db_file}' silindi.")
    
    print("Database Module Test Başlatılıyor...")
    if not init_db():
        print("DB başlatılamadı, testler durduruldu.")
        exit()
    
    # Örnek veriler ekle
    print("\nÖrnek plakalar ekleniyor...")
    add_plate("34ABC111", "truck", True)
    add_plate("34DEF222", "car", False)
    add_plate("34GHI333", "truck", False)
    add_plate("34ABC111", "truck", True) # Tekrar eklemeyi dene (başarısız olmalı)

    print("\nTüm Plakalar (Filtresiz):")
    for p in get_all_plates():
        print(f"  ID: {p['id']}, Plaka: {p['plate_number']}, Tür: {p['vehicle_type']}, Yetki: {p['is_authorized']}, Tarih: {p['added_date']}")

    print("\n'ABC' içeren plakalar:")
    for p in get_all_plates(search_term="ABC"):
        print(f"  ID: {p['id']}, Plaka: {p['plate_number']}, Tarih: {p['added_date'].strftime('%d-%m-%Y %H:%M') if p['added_date'] else 'Yok'}")
    
    print("\nSadece Kamyonlar:")
    for p in get_all_plates(vehicle_filter="truck"):
        print(f"  ID: {p['id']}, Plaka: {p['plate_number']}")

    # Örnek aktivite logları
    log_activity("KAMYON_GIRIS_YETKILI", "34ABC111", "truck", "static/log_images/sample_truck.jpg", "{'confidence':0.9}")
    log_activity("OTOMOBIL_YETKISIZ", "34DEF222", "car", None, "{'reason':'Policy'}")

    print("\nSon Aktivite Logları:")
    for log in get_activity_logs(5):
        print(f"  ID: {log['id']}, Olay: {log['event_type']}, Plaka: {log.get('plate_number', '-')}, Zaman: {log.get('timestamp').strftime('%Y-%m-%d %H:%M') if log.get('timestamp') else 'Yok'}")

    print("\nDatabase Module Test Tamamlandı.")