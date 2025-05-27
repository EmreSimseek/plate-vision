// PlateVision/static/js/script.js

// Anlık durumu backend'den çekip güncelleyen fonksiyon
function fetchStatus() {
    const statusArea = document.getElementById('status-area');
    if (!statusArea) return; // Eğer status-area elementi sayfada yoksa bir şey yapma

    fetch('/get_status') // Flask'ta bu endpoint'i oluşturacağız
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                statusArea.textContent = data.text_message || 'Durum bilgisi alınamadı.';
                // CSS sınıfını güncelle
                statusArea.className = 'status-box'; // Önceki sınıfları temizle
                if (data.message_class) {
                    statusArea.classList.add(data.message_class);
                } else {
                    statusArea.classList.add('neutral');
                }
            }
        })
        .catch(error => {
            console.error('Durum bilgisi alınırken hata oluştu:', error);
            statusArea.textContent = 'Sunucuyla bağlantı kurulamadı veya bir hata oluştu.';
            statusArea.className = 'status-box danger';
        });
}

// Sayfa yüklendiğinde ve belirli aralıklarla durumu güncellemek için
// Bu, index.html içindeki script bloğundan çağrılabilir veya burada doğrudan çalıştırılabilir.
// index.html'de zaten bir `setInterval` var, o yüzden burada tekrar çalıştırmaya gerek yok
// eğer `index.html`'deki `DOMContentLoaded` bloğu bu fonksiyonu çağırıyorsa.
// Eğer index.html'de interval yoksa, aşağıdaki satırı açabilirsiniz:
// document.addEventListener('DOMContentLoaded', function() {
//    if (document.getElementById('status-area')) { // Sadece ana sayfada çalışsın
//        setInterval(fetchStatus, 1000); // Her 1 saniyede bir durumu al
//    }
// });

// Genel diğer JavaScript fonksiyonları buraya eklenebilir.
console.log("PlateVision script.js yüklendi.");