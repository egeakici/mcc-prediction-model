# İşletme adından MCC: ilk TF-IDF deneyi

Amaç, işletme adı metninin MCC kodu hakkında taşıdığı bilgiyi kaynaklar arası,
merchant gruplarını ayırarak ölçmektir. Başarıyı artırmak için veri bölmesi veya
etiket uzayı test sonuçlarına göre değiştirilmez. Otomatik rapor yalnızca tablo
ve sayılardan oluşur.

## Kurulum ve tam çalıştırma (PowerShell)

Python 3.13 ile doğrulanmıştır. Depoyu indirdikten sonra proje klasöründe:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Doğrudan bağımlılıklar sabit sürümlüdür. Windows/Python 3.13 ortamının geçişli
bağımlılıkları dahil tam dökümü `requirements-lock.txt` içindedir; ortamı birebir
kurmak için `python -m pip install -r requirements-lock.txt` kullanabilirsiniz.

Tam deney için komutları sırayla çalıştırın:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.build_table
python -m src.make_splits
python -m src.train_tfidf
python -m src.evaluate
python -m src.leakage_check
python -m src.report
```

`leakage_check` isteğe bağlıdır; atlarsanız rapor mevcut diğer tabloları üretir.
PowerShell etkinleştirme politikanız aktivasyonu engelliyorsa komutlardaki
`python` yerine `.\.venv\Scripts\python.exe` kullanabilirsiniz.

Veri hazırlığı ve eğitim ayrı terminal komutlarıdır; bağımlılık kurulumu eğitimi
başlatmaz.

## Veri dosyaları

Veri dosyaları Git deposuna dahil değildir. Deneyi yeniden çalıştırmak için
aşağıdaki dosyaları kendi yerel kopyalarınızdan sağlamalısınız:

```text
data/
  MANIFEST.md
  raw/
    oklahoma.parquet
    delaware.parquet
    sunderland.parquet
  reference/
    mcc_desc_to_code.csv
    mcc_reference_snapshot.csv
```

Ham dosyalarda beklenen sütunlar `configs/config.yaml` içindeki
`required_columns` alanında tanımlıdır. Manifest veri kaynağını, hazırlama
adımlarını ve dosya kimliklerini belgelemelidir. `data/raw` ve `data/reference`
hattın salt okunur girdileridir. Referans Excel dosyası bu deneyde kullanılmaz.

## Tam deney sonuçları

Model Oklahoma'da eğitildi; sınıf uzayı 233 MCC kodu, model girdisi yalnızca
normalize edilmiş işletme adı metnidir. C değeri doğrulama kümesinde 0,3 seçildi.

| Test kaynağı | İşlem doğruluğu | Merchant doğruluğu | İlk 3 doğruluğu | Macro-F1 |
|---|---:|---:|---:|---:|
| Oklahoma | %58,65 | %42,98 | %80,53 | 0,3653 |
| Delaware | %61,82 | %46,37 | %76,71 | 0,2987 |
| Sunderland | %35,04 | %34,35 | %49,53 | 0,4100 |

İlk 3 doğruluğu işlem ağırlıklıdır. Macro-F1, testte en az beş merchant'ı
olan sınıflarda hesaplanır; sınıf sayısı sırasıyla 175, 120 ve 16'dır.
Sunderland testi 294 merchant içerir ve işlem doğruluğunun güven aralığı
geniştir. Bütün test etiketleri açıklamadan bağlanan (`linked`) etiketlerdir.

Güven aralıkları ve hata tabloları [tam raporda](outputs/reports/report_tfidf.md),
sayısal değerler [results.csv](outputs/metrics/results.csv) dosyasındadır.

## Duman testi

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.build_table --sample 20000
python -m src.make_splits --sample 20000
python -m src.train_tfidf --sample 20000
python -m src.evaluate --sample 20000
python -m src.leakage_check --sample 20000
python -m src.report --sample 20000
python -m unittest discover -s tests -v
```

Veri ve örneklem çıktıları bulunmayan yeni bir klonda da birim testleri
`python -m unittest discover -s tests -v` ile çalışır. Örneklem çıktılarına bağlı
test sınıfı, bu dosyalar üretilene kadar atlanır. Testler eğitim başlatmaz.

Örneklem hattı her kaynaktan sabit tohumla, yerine koymadan rastgele N ham satır
seçer; bu duman testi bir performans tahmini olarak kullanılmaz. Sızıntı deneyi ise
Oklahoma'nın dışlamalar sonrası **tekilleştirilmemiş tüm ham satırlarından**,
sabit tohumla rastgele 50.000 satır seçer. Burada `--sample 20000`, önceki
örneklem modelini/C seçimini ve ayrı çıktı alanını tanımlar; sızıntı deneyi için
20.000 satır sınırı koymaz. Tam deneyde bu sayı 1.000.000'dur. Her ikisi YAML'da
tanımlıdır. Gerekirse yalnızca bu sayı `--rows N` ile geçersiz kılınabilir.

## Yapılandırma ve korumalar

Yollar, dosya adları, eşikler, tohum, fold rolleri ve hiperparametreler
`configs/config.yaml` dosyasındadır. Bütün betikler `--config PATH` destekler.
Göreli yollar config dosyasının üst klasörünün üstüne (proje köküne) göre çözülür.
CLI'nin varsayılan config konumu tek başlangıç yoludur.

`data/raw` ve `data/reference` yalnızca okunur. Çıktı yolu koruması bu klasörlere
ve proje dışına yazılmasını reddeder. Excel taksonomi dosyası kullanılmaz.
Manifest her komutta veri okumadan önce okunur; Parquet şemaları doğrulanır.
Eksik zorunlu sütun, belirsiz referans eşlemesi, geçersiz etiket veya boş
`key_loose` varsa betik durur; tasarım eşikleri kendiliğinden değiştirilmez.

Örneklem çıktıları `outputs/interim_sample`, `splits_sample`, `models_sample`,
`predictions_sample`, `metrics_sample`, `reports_sample`, `logs_sample` altında
üretilir. Tam hat `data/interim`, `data/splits` ve `outputs` altındaki normal
klasörleri kullanır. Farklı N/config ile oluşturulmuş ara çıktılar karıştırılmaz;
önceki adımları yeniden üretmek gerekir.

`splits.parquet` tekrar çalıştırmada üzerine yazılmaz. Mevcut dosya, metadata ve
girdi parmak izleri kontrol edilerek korunur. Açıkça yeniden üretmek için:

```powershell
python -m src.make_splits --force
# Örneklem için:
python -m src.make_splits --sample 20000 --force
```

YAML değiştirilirse önceki aşamaları yeniden çalıştırın; bölmeleri değiştirmek
gerekiyorsa `--force` kullanın. Aynı deneyde örneklem komutlarının tümünde aynı
`--sample N` kullanılmalıdır. Günlükler başlangıç/bitiş sayıları, süreler,
eğitimden çıkarılan sınıf/satır sayıları ve yakınsama uyarılarını saklar.
Başlangıçta bloklanan lazy/yerel işlemlerde tqdm geçen süreyi gösterir ve sonuç
geldiğinde sayıyı günceller; sınıf, batch, fold ve bootstrap döngülerinde canlı
tamamlanma ilerlemesi gösterilir.

## Deney tanımları

- Dışlamalar sırayladır; `exclusions.csv` sayıları çifte sayılmaz. Kodsuz satırlar,
  önceki record_type/mask filtrelerinden sonra orijinal sütunlarıyla ayrı kaydedilir.
  Özetin `excluded_rows` alanı model tablosuna girmeyen bütün satırları, kodsuzlar
  dahil, sayar. Etiket kodları string kalır; dört haneye sıfırla tamamlanır.
- Normalizasyon yalnızca küçük harfe çevirme, boşluk sıkıştırma ve kırpmadır.
  `add_intermediary: false`. `key_loose` değiştirilmeden SHA-1'e girer.
- Oklahoma fold 0–7 eğitim, 8 doğrulama, 9 testtir. Diğer kaynakların yalnızca
  fold 9'u testtir; diğer fold'lar dosyada `unused` olarak korunur. `seen_in_ok`
  Oklahoma'nın tüm uygun fold'ları üzerinden hesaplanır; modele verilmez.
- Sınıf uygunluğu eğitimde **tekil text sayısı** ile belirlenir. OVR'da her model
  tablosu kaydının ağırlığı birdir; `n_rows` kullanılmaz. Majority aynı uygun
  eğitim sınıflarında tekil text sayısına göre sıralama yapar; top-1 en sık kod,
  top-3 bu sıralamanın ilk üç kodudur. Eşit sayıda kodda leksikografik sıra kullanılır.
- C, tüm validation kayıtlarındaki ağırlıksız macro-F1 ile seçilir; OOV gerçek
  sınıflar bu skora dahildir. Eşit skorda YAML C listesindeki ilk değer seçilir.
  Seçimden sonra val eğitime eklenmez. `tune_c: false` iken `default_c` kullanılır.
- `acc_texts`, tekil `(text, label_code)` çiftlerini birer kez sayar. Merchant
  gerçek etiketi, key_loose içindeki **etikete ait toplam n_rows** ile seçilir.
  Tahmin için aynı key_loose içindeki **metne ait toplam n_rows** en büyük olan
  text kullanılır. Köken/etiket yüzünden parçalanmış kayıtlar önce toplanır.
  Etiket ve metin eşitlikleri leksikografik artan sıra ile çözülür.
- Macro-F1 ve sınıf raporlarının destekleri merchant düzeyindedir. Macro-F1,
  testte en az beş merchant destekli gerçek sınıfları kullanır. Bu sınıf listesi
  bootstrap boyunca sabittir; yeniden örneklemede destek eşiği tekrar uygulanmaz.
  Her bootstrap örneklemi n merchant'ı yerine koyarak seçer; bir merchant'ın bütün
  satırları aynı katsayıyla taşınır. Percentile %95 aralık, 1.000 tekrar, tohum 42.
- Hata paylarının paydası yalnızca yanlış ve eğitim sınıf uzayında olan gerçek
  etiketlerin satır toplamıdır. Payda sıfırsa değer boştur. Diğer tanımsız ölçüler
  ve boş testler de boş değerle raporlanır. OOV örnekler testten çıkarılmaz.
- Sızıntı kontrolü aynı sınıf eşiğini, vektörleştiricileri, OVR ayarlarını ve
  seçilmiş C'yi kullanır; vektörleştiriciler o deneyin %80 eğitim kısmında fit edilir.
  Gruptan bağımsız satır bölmesi ile merchant bölmesi ayrı protokoller olarak
  yan yana gösterilir; acc_rows ve acc_merchant ayrı sütunlarda kalır.
- Hata tabloları her model/test için aynı metin/gerçek/tahmin çiftlerinin n_rows
  toplamına göre ilk 15 kaydı verir. Rapor sayıları [0,1] ölçeğinde gösterir.

## Üretilen dosyalar

Kaynak kodu: `src/config.py`, `text_norm.py`, `build_table.py`, `make_splits.py`,
`train_tfidf.py`, `evaluate.py`, `leakage_check.py`, `report.py`, `__init__.py`.

Ara çıktılar: `model_table.parquet`, `no_code_<source>.parquet`, `exclusions.csv`,
`table_summary.csv`, `build_info.json`; bölmeler: `splits.parquet`, `split_info.json`.

Modeller: `tfidf_linearsvc/model.joblib`, `train_info.json`. Tek joblib dosyası
iki vektörleştiriciyi, sıralı sınıfları, katsayı matrisini, intercept vektörünü
ve majority sıralamasını içerir.

Değerlendirme: altı `<model>_<test>.parquet`, `results.csv`, `c_search.csv`, altı
`per_class_<model>_<test>.csv`, isteğe bağlı `leakage.csv`.

Rapor: `report_tfidf.md`. Günlükler: her aşama için `<module>.log` ve
`<module>_config.yaml`. Sürümler ve girdi SHA-256 değerleri metadata'dadır.

Uygulama API referansları: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
[joblib generator](https://joblib.readthedocs.io/en/stable/auto_examples/parallel_generator.html).

## Git deposunun kapsamı

Depoda kaynak kodu, YAML ayarları, testler, bağımlılık sürümleri, README ve tam
deneyin metrik CSV'leri/Markdown raporu tutulur. Rapor hata örneklerinde işletme
adları içerir. Tablolar, deney yeniden çalıştırılırsa güncellenebilir.

`.gitignore` şu dosyaları yerelde bırakır:

| Kural | Kapsam ve gerekçe |
|---|---|
| `/data/` | Ham veriler, referanslar, manifest, ara tablolar ve bölmeler |
| `/outputs/*` | Modeller, tahminler, günlükler, joblib geçici dosyaları ve bütün duman testi çıktıları |
| `!/outputs/metrics/*.csv` | İstisna: tam deneyin küçük metrik CSV dosyaları depoya girer |
| `!/outputs/reports/*.md` | İstisna: tam deneyin Markdown raporu depoya girer |
| `/.venv/`, `/venv/` | Yeniden kurulabilen yerel Python ortamı |
| `__pycache__/`, `*.py[cod]`, araç önbellekleri | Üretilen Python/araç dosyaları |
| `*.parquet`, `*.joblib`, `*.pkl`, `*.pickle` | Yanlış klasöre kaydedilmiş büyük veri/model dosyaları için ek koruma |
| `/.tmp/`, `*.log`, `/DELIVERY.md` | Geçici dosyalar, günlükler ve bu çalışma ortamına ait teslim notları |
| `.env*`, `*.pem`, `*.key`, `/.aws/` | Yerel sırlar ve kimlik bilgileri; `.env.example` istisnadır |
| `/.codex/`, `/.agents/`, `/.idea/`, `/.vscode/` | Yerel araç ve editör ayarları |

Ignore kuralları dosyaları diskten silmez. Git boş klasörleri izlemez;
gerekli çıktı klasörlerini betikler oluşturur. `.gitattributes`, kaynak metin
dosyalarının Git'te LF satır sonlarıyla saklanmasını sağlar.

İlk yerel commit için (henüz Git deposu yoksa):

```powershell
git init -b main
git add .gitattributes .gitignore README.md requirements.txt requirements-lock.txt configs src tests outputs/metrics outputs/reports
git diff --cached --stat
git commit -m "Add reproducible TF-IDF MCC experiment and results"
```

Sonraki değişikliklerde:

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "Describe the experiment changes"
```
