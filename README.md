# Hibrit ADAS Projesi

Bu proje, YOLOPv2 ile şerit/sürüş alanı segmentasyonu ve YOLO11s ile nesne
tespitini aynı video akışında birleştirir. Mesafe, nesnenin yaklaşık gerçek
yüksekliği ve görüntüdeki piksel yüksekliği kullanılarak tahmin edilir.

## Çalıştırma

Gerekli dosyalar aynı klasörde bulunmalıdır:

- `test.mp4`
- `yolopv2.pt`
- `yolo11s.pt`

```powershell
python hybrid_adas.py --input test.mp4 --output test_hibrit.mp4
```

Kısa test için:

```powershell
python hybrid_adas.py --max-frames 100
```

Notebook, deney ve görselleştirme için korunmuştur. Tekrarlanabilir ana
pipeline ise `hybrid_adas.py` dosyasındadır.
