# Jawaban — A.1 Analisis `broken_training_step()`

> **Soal(bagian "A. Analisis Kode", No. 1)**
>
> ```python
> def broken_training_step(model, optimizer, criterion, X, y):
>     model.eval()
>     optimizer.zero_grad()
>     prediction = model(X)
>     loss = criterion(prediction, y)
>     loss.backward()
>     optimizer.step()
>     return loss.item()
> ```
>
> Apa masalah utama dari penggunaan `model.eval()` di dalam fungsi tersebut? Jelaskan:
> - Fungsi `model.eval()`.
> - Mode yang seharusnya digunakan saat training.
> - Jenis layer yang perilakunya dapat berubah antara training dan evaluation.
> - Mengapa masalah ini dapat memengaruhi hasil training.

---

## Jawaban
### Fungsi `model.eval()`

`model.eval()` hanya mengubah flag training menjadi `False` pada model dan semua submodulnya (sama dengan `model.train(False)`). Jadi di `broken_training_step`, bobot tetap diperbarui, tetapi beberapa layer berperilaku dalam mode evaluasi.

### Mode yang seharusnya dipakai saat training

`model.train()`. Pola standarnya: `model.train()` sebelum training, `model.eval()` + `torch.no_grad()` saat evaluasi.

### Layer yang perilakunya berubah

- `Dropout` : aktif saat train, menjadi identitas saat eval.
- `BatchNorm` : saat train memakai statistik batch dan memperbarui `running_mean` / `running_var`, saat eval memakai running statistics.
- `InstanceNorm` dengan `track_running_stats=True`.
- `RNN`/`LSTM`/`GRU` dengan argumen `dropout` antar-layer.

### Mengapa memengaruhi hasil training

- Dropout mati, sehingga tidak ada regularisasi dan model lebih mudah overfit.
- BatchNorm memakai running stats yang tidak pernah diperbarui (tetap nilai awal mean 0, var 1). Layer ini praktis tidak menormalisasi apa pun, dan statistik data tidak pernah dipelajari.
- Bug ini bisa bersifat laten. Model di kode Anda hanya berisi `Linear` dan `ReLU`, jadi tidak ada perbedaan numerik. Bug baru terasa saat Anda menambah Dropout atau BatchNorm. Eksperimen 2 membuktikannya.

# Jawaban — A.2 Perbaikan Kode (`corrected_training_step()`)

> **Soal (bagian "A. Analisis Kode", No. 2)**
>
> Ubahlah fungsi `broken_training_step()` sehingga menggunakan mode yang sesuai untuk proses training. Tuliskan versi kode yang telah diperbaiki dan jelaskan alasan perubahan tersebut.
>
> Petunjuk: `model.train()`

---

## Kode yang Diperbaiki

```python
def corrected_training_step(model, optimizer, criterion, X, y):
    model.train()          # <- perbaikan: gunakan train mode, bukan eval mode
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Satu-satunya perubahan dari `broken_training_step()` adalah baris pertama: `model.eval()` → `model.train()`. Sisa urutan (`zero_grad -> forward -> loss -> backward-> step`) sudah benar dan tidak perlu diubah karena itu adalah *training loop* standar, masalahnya murni pada mode model.

## Alasan Perubahan

### 1. model.eval() tidak melindungi proses training

Orang sering mengira `eval()` "mengamankan" model dari perubahan yang tidak diinginkan. Faktanya:

`requires_grad` semua parameter tetap `True`
`.grad` tetap terisi setelah `backward()`

Artinya `optimizer.step()` di `broken_training_step()` tetap benar-benar mengubah bobot, sama seperti versi corrected. Jadi`eval()` di sini tidak mencegah training terjadi — ia cuma mengubah bagaimana layer tertentu berperilaku selama forward pass, sambil training tetap berjalan seolah-olah tidak ada masalah. Ini yang membuat bug jenis ini berbahaya: tidak ada error, tidak ada crash, training "terlihat" jalan normal.

### 2. Dropout mati total selama training jika pakai `eval()`

Pada `Dropout(p=0.5)` dengan input yang sama:

mode train: sebagian nilai di-nol-kan secara acak, sisanya diskalakan naik (×2)
mode eval: fungsi identitas, semua nilai lolos apa adanya

Kalau `broken_training_step()` dipakai pada model yang punya `Dropout`, setiap forward pass selama training akan berjalan di mode eval, artinya dropout tidak pernah aktif sepanjang training. Padahal dropout adalah mekanisme regularisasi yang sengaja dipasang untuk mencegah neuron saling "bergantung" berlebihan.

### 3. `BatchNorm` tidak pernah belajar statistik data yang sebenarnya

Setelah membandingkan `running_mean`/`running_var` setelah 20 iterasi:

training di mode eval: statistik tetap di nilai default (`mean 0, var 1`), `num_batches_tracked` tetap 0
training di mode train: statistik bergerak mendekati distribusi data asli (`mean ~10, var ~9`)

Ini karena BatchNorm hanya meng-update `running_mean`/`running_var` saat `self.training == True`. Kalau training dijalankan di mode eval, BatchNorm memakai statistik batch normalisasi versi eval (running stats yang belum pernah diperbarui), sehingga normalisasi tidak pernah menyesuaikan diri dengan data sebenarnya sehingga fungsi utama BatchNorm gagal total, bukan cuma berkurang efektivitasnya.

# Jawaban — A.3 Apakah `model.eval()` Menghentikan Gradient?

> **Soal (bagian "A. Analisis Kode", No. 3)**
>
> Benar atau salah: "Pemanggilan `model.eval()` secara otomatis menghentikan perhitungan gradient dan membuat parameter model tidak dapat diperbarui."
>
> Jelaskan dengan membedakan `model.eval()` vs `torch.no_grad()`.

---

## Jawaban: **Salah**

`model.eval()` **tidak** menghentikan perhitungan gradient dan **tidak** membuat parameter model tidak dapat diperbarui. Ini adalah kesalahpahaman umum, justru karena namanya terdengar seperti mekanisme untuk mematikan gradient padahal bukan.

## Mengapa: `model.eval()` vs `torch.no_grad()` mengontrol hal yang benar-benar berbeda

### 1.`model.eval()`

`model.eval()` adalah method yang mengubah atribut `.training` pada model dan seluruh submodulnya menjadi `False`. Fungsinya murni untuk memberi tahu layer-layer tertentu bahwa mereka sedang tidak dalam fase training, sehingga layer seperti `Dropout` dan `BatchNorm` beralih ke perilaku evaluasi: Dropout berhenti menonaktifkan neuron secara acak dan menjadi fungsi identitas, sedangkan BatchNorm berhenti menghitung statistik dari batch yang sedang berjalan dan menggantinya dengan running mean/variance yang telah terkumpul selama training. Mekanisme ini tidak berkaitan sama sekali dengan gradient atau computation graph — ia hanya sebuah saklar perilaku (behavior switch) untuk layer-layer yang sensitif terhadap mode, dan tidak memengaruhi apakah proses backward atau update bobot dapat berjalan.

### 2.`torch.no_grad()`

`torch.no_grad()` adalah context manager yang menonaktifkan pencatatan operasi ke dalam computation graph milik autograd selama blok kode tersebut dijalankan. Ketika aktif, setiap tensor hasil operasi di dalamnya memiliki `requires_grad=False` dan `grad_fn=None`, sehingga PyTorch tidak menyimpan riwayat operasi yang dibutuhkan untuk menghitung gradient, dan `loss.backward()` tidak dapat dipanggil atas tensor tersebut. Tujuan utamanya adalah efisiensi: karena saat evaluasi atau inference gradient memang tidak diperlukan, menonaktifkan pencatatan graph ini menghemat memori dan mempercepat komputasi secara signifikan, tanpa mengubah sedikit pun perilaku layer seperti Dropout atau BatchNorm. 

# Jawaban — A.4 Urutan Training Step

> **Soal (bagian "A. Analisis Kode", No. 4)**
>
> Jelaskan fungsi dan urutan operasi berikut:
>
> ```python
> optimizer.zero_grad()
> prediction = model(X)
> loss = criterion(prediction, y)
> loss.backward()
> optimizer.step()
> ```

---

## Jawaban

| Operasi | Fungsi (dari PyTorch docs) |
|---|---|
| `optimizer.zero_grad()` | `torch.optim.Optimizer.zero_grad()` "Resets the gradients of all optimized `torch.Tensor`s." Secara default, gradient di PyTorch **terakumulasi** (accumulate) setiap kali `.backward()` dipanggil, jadi `.grad` dari setiap parameter perlu di-reset ke nol (atau `None`) sebelum backward pass berikutnya, supaya gradient dari batch sebelumnya tidak ikut tercampur. |
| `model(X)` (forward pass) | Memanggil `nn.Module.forward()` model (lewat `__call__`). Menurut *Autograd mechanics* PyTorch: "During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad." Jadi selain menghasilkan `prediction`, langkah ini juga membangun **computation graph** yang akan dipakai untuk menghitung gradient di backward pass. |
| `criterion(prediction, y)` | Memanggil loss function (di sini `nn.BCEWithLogitsLoss`), yang menghitung satu nilai skalar (`loss`) yang mengukur seberapa jauh `prediction` dari label `y`. Nilai skalar ini menjadi titik awal (root) dari graph yang akan di-*backward*-kan. |
| `loss.backward()` | `torch.Tensor.backward()` "Computes the gradient of current tensor wrt graph leaves." Ini menjalankan **backpropagation**: menelusuri computation graph dari `loss` mundur ke semua leaf tensor yang memiliki `requires_grad=True` (yaitu parameter model), dan mengakumulasikan hasilnya ke `.grad` masing-masing parameter. Sesuai kutipan *Autograd mechanics* sebelumnya: "During the backward pass (`.backward()`), only leaf tensors with `requires_grad=True` will have gradients accumulated into their `.grad` fields." |
| `optimizer.step()` | `torch.optim.Optimizer.step()` "Performs a single optimization step (parameter update)." Menggunakan nilai `.grad` yang sudah dihitung oleh `loss.backward()` untuk memperbarui parameter model sesuai algoritma optimizer yang dipakai (di sini `Adam`) dan learning rate yang ditentukan. |

## Mengapa urutannya harus seperti itu

Urutan ini bukan kebetulan setiap langkah *bergantung* pada hasil langkah sebelumnya:

1. **`zero_grad()` harus di awal** (sebelum `backward()` berikutnya) karena gradient bersifat akumulatif jika tidak di-reset, gradient dari step sebelumnya akan tercampur dengan gradient step ini (lihat A.5).
2. **`model(X)` harus sebelum `criterion(...)`** karena loss function butuh `prediction` sebagai input.
3. **`criterion(...)` harus sebelum `backward()`** karena yang di-*backward*-kan adalah `loss` (skalar), bukan `prediction` dan `backward()` dipanggil dari titik akhir graph.
4. **`backward()` harus sebelum `step()`** karena `optimizer.step()` membaca `.grad` yang baru terisi setelah `backward()` dijalankan; kalau dipanggil sebelum `backward()`, `.grad` masih kosong/dari step sebelumnya sehingga update parameter tidak sesuai gradient loss saat ini.

Pola ini adalah training loop standar di PyTorch: `zero_grad → forward → loss → backward → step`, dan tidak ada langkah yang bisa ditukar urutannya tanpa merusak logika training.

# Jawaban — A.5 Risiko Melupakan `zero_grad()`

> **Soal (bagian "A. Analisis Kode", No. 5)**
>
> Apa yang dapat terjadi jika baris `optimizer.zero_grad()` dihilangkan? Jelaskan mengapa gradient dapat terakumulasi dari beberapa mini-batch dan bagaimana hal tersebut dapat memengaruhi proses pembelajaran.

---

## Jawaban

### Apa yang terjadi jika baris tersebut dihilangkan?
 
Jika `optimizer.zero_grad()` dihapus dari training step, gradient dari setiap
mini-batch **tidak akan pernah direset**. Setiap kali `loss.backward()`
dipanggil, PyTorch tidak menimpa nilai `.grad` yang sudah ada, melainkan
**menjumlahkannya (akumulasi)** ke nilai `.grad` sebelumnya. Akibatnya,
`optimizer.step()` akan memperbarui bobot menggunakan gradient yang sudah
tercampur dari banyak mini-batch sebelumnya, bukan gradient murni dari batch
yang sedang diproses.
 
### Mengapa gradient bisa terakumulasi antar mini-batch?
 
Perilaku ini bukan bug, melainkan desain PyTorch yang disengaja. Secara
default, setiap kali `.backward()` dipanggil pada sebuah tensor, PyTorch
menjalankan:
 
```python
param.grad = param.grad + gradien_baru   # jika param.grad sudah ada
param.grad = gradien_baru                # jika param.grad masih None
```
 
Desain ini sengaja dibuat seperti itu agar mendukung kasus penggunaan
tertentu, misalnya:
 
- **Gradient accumulation** — mensimulasikan batch size besar dengan
  menjumlahkan gradient dari beberapa mini-batch kecil sebelum memanggil
  `optimizer.step()`, biasanya karena keterbatasan memori GPU.
- Menghitung gradient dari beberapa loss terpisah (multi-task) sebelum satu
  kali update.
Tanggung jawab **kapan** gradient direset diserahkan sepenuhnya ke
pengguna melalui `optimizer.zero_grad()`. Kalau baris ini tidak dipanggil di
setiap iterasi, gradient dari batch 1 akan ikut terjumlah ke batch 2, batch 2
ke batch 3, dan seterusnya — terus menumpuk sepanjang training tanpa pernah
kembali ke nol.
 
### Bagaimana hal ini memengaruhi proses pembelajaran?
 
1. **Skala gradient membesar seiring waktu.** Setelah N batch tanpa reset,
   gradient yang dipakai untuk update kira-kira adalah jumlah dari N gradient
   batch, bukan gradient dari satu batch. Ini secara efektif membuat langkah
   update jauh lebih besar dari yang seharusnya, seolah-olah learning rate
   membesar terus-menerus.
2. **Update bobot tidak lagi mencerminkan batch saat ini.** Arah update
   seharusnya dipandu oleh error pada mini-batch yang sedang diproses, tapi
   karena tercampur dengan gradient batch-batch lama (yang mungkin sudah
   tidak relevan setelah bobot berubah), arah pembaruan menjadi bias dan
   tidak konsisten dengan kondisi model terkini.
3. **Training menjadi tidak stabil atau divergen.** Karena magnitudo
   gradient terus bertambah, loss bisa berosilasi liar, meledak
   (`NaN`/`inf`), atau model gagal konvergen sama sekali, terutama semakin
   lama training berjalan karena akumulasi terus bertumpuk.
4. **Sulit didiagnosis.** Sama seperti kasus `model.eval()` yang keliru,
   kode tetap berjalan tanpa error. Training hanya terlihat "aneh" — loss
   naik-turun tidak wajar atau tidak turun sama sekali — sehingga
   penyebabnya mudah disalahartikan sebagai masalah lain (learning rate,
   arsitektur, data) padahal akar masalahnya adalah gradient yang tidak
   pernah direset.

# Jawaban — B.6 Analisis Shape

> **Soal (bagian "B. Pemeriksaan Input dan Output", No. 6)**
>
> Berdasarkan kode program, tentukan:
> 1. Shape `X`.
> 2. Shape `y`.
> 3. Shape output model.
> 4. Jumlah fitur input.
> 5. Jumlah output model.
>
> Jelaskan mengapa shape label dibuat seperti berikut:
> ```python
> y = torch.randint(0, 2, (16, 1)).float()
> ```

---

## Jawaban

Berdasarkan kode di bagian `__main__`:

```python
X = torch.randn(16, 2)
y = torch.randint(0, 2, (16, 1)).float()
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
```

### 1. Shape `X`

`(16, 2)` — 16 sampel (baris), masing-masing dengan 2 fitur (kolom). Dimensi pertama pada tensor input selalu diperlakukan sebagai dimensi batch oleh layer `nn.Linear`.

### 2. Shape `y`

`(16, 1)` — 16 label, masing-masing berupa satu nilai skalar (0 atau 1) yang dibungkus dalam dimensi kedua berukuran 1.

### 3. Shape output model

`(16, 1)`. Ini bisa ditelusuri dari arsitektur:
- `nn.Linear(2, 8)` mengubah `(16, 2)` → `(16, 8)`
- `nn.ReLU()` tidak mengubah shape, tetap `(16, 8)`
- `nn.Linear(8, 1)` mengubah `(16, 8)` → `(16, 1)`

### 4. Jumlah fitur input

2, sesuai argumen pertama `nn.Linear(2, 8)`, yang juga cocok dengan dimensi kedua dari `X` yaitu `(16, 2)`.

### 5. Jumlah output model

1, sesuai argumen kedua `nn.Linear(8, 1)`. Ini karena kasusnya adalah klasifikasi biner (0 atau 1), sehingga cukup satu logit per sampel — probabilitas kelas positif bisa diturunkan dari satu nilai ini (lihat B.9).

## Mengapa shape label dibuat `(16, 1)`, bukan `(16,)`?

Alasan utamanya adalah **agar shape label sama persis dengan shape output model**, yaitu `(16, 1)`. Ini penting karena `nn.BCEWithLogitsLoss()` (dan sebagian besar loss function PyTorch) membandingkan `prediction` dan `y` secara element-wise, sehingga kedua tensor idealnya memiliki shape yang identik.

Jika `y` dibiarkan berbentuk `(16,)` sedangkan `prediction` berbentuk `(16, 1)`, PyTorch akan melakukan **broadcasting** antara shape `(16,)` dan `(16, 1)` menjadi `(16, 16)` alih-alih error langsung. Ini adalah bug diam-diam (silent bug) yang sangat umum: loss tetap terhitung, tidak ada error, tetapi nilainya salah karena setiap prediksi dibandingkan dengan seluruh 16 label, bukan hanya label pasangannya. Dengan membentuk `y` sebagai `(16, 1)` sejak awal (melalui argumen shape kedua `torch.randint(0, 2, (16, 1))`), masalah broadcasting ini dihindari sepenuhnya.

---

# Jawaban — B.7 Analisis Dtype

> **Soal (bagian "B. Pemeriksaan Input dan Output", No. 7)**
>
> Mengapa label diubah menjadi tipe floating-point?
> ```python
> y = torch.randint(0, 2, (16, 1)).float()
> ```
> Jelaskan hubungan antara dtype label dan loss function berikut:
> ```python
> criterion = nn.BCEWithLogitsLoss()
> ```

---

## Jawaban

### Mengapa label diubah menjadi floating-point?

`torch.randint(0, 2, (16, 1))` secara default menghasilkan tensor dengan dtype integer (`torch.int64`), karena fungsi ini dirancang untuk menghasilkan bilangan bulat acak. Namun, `.float()` dipanggil setelahnya untuk mengubah dtype tersebut menjadi `torch.float32`, karena loss function yang dipakai — `nn.BCEWithLogitsLoss()` — mewajibkan input dan target bertipe floating-point, bukan integer.

### Hubungan dtype label dengan `BCEWithLogitsLoss()`

`nn.BCEWithLogitsLoss()` menghitung *binary cross-entropy* dengan menggabungkan operasi sigmoid dan BCE loss dalam satu langkah yang stabil secara numerik. Secara internal, fungsi ini melakukan operasi floating-point kontinu, antara lain menghitung `log(sigmoid(x))`, yang mengharuskan baik `prediction` maupun `target` bertipe `float` (umumnya `float32`).

Jika `y` dibiarkan bertipe `int64` (tanpa `.float()`), PyTorch akan melempar `RuntimeError` karena dtype `prediction` (float, hasil dari `nn.Linear`) dan `target` (int) tidak cocok — operasi elemen-per-elemen dalam loss function ini menuntut kedua tensor berada dalam tipe numerik yang sama (floating-point).

Selain itu, secara konseptual label 0/1 di sini bukan diperlakukan sebagai kategori diskrit seperti pada `CrossEntropyLoss` (yang justru menuntut target berupa `int64`/`long`), melainkan sebagai **target probabilitas** (0.0 atau 1.0) yang dibandingkan langsung dengan probabilitas hasil sigmoid dari logit model. Karena itu representasi float adalah representasi yang benar secara matematis untuk `BCEWithLogitsLoss`, bukan sekadar syarat teknis.

---

# Jawaban — B.8 Analisis Device

> **Soal (bagian "B. Pemeriksaan Input dan Output", No. 8)**
>
> Mengapa model dan tensor input harus berada pada device yang sama? Berikan contoh masalah yang dapat muncul apabila model berada di GPU dan `X` berada di CPU. Bagaimana cara memindahkan tensor atau model ke device yang sesuai?

---

## Jawaban

### Mengapa model dan tensor input harus berada di device yang sama?

Setiap operasi tensor di PyTorch (misalnya perkalian matriks pada `nn.Linear`) dijalankan oleh kernel komputasi yang spesifik untuk satu device tertentu — kernel CPU tidak bisa langsung beroperasi pada memori GPU, begitu pula sebaliknya. Bobot model (parameter) dan tensor input harus berada di ruang memori fisik yang sama agar operasi seperti `X @ W.T + b` dapat dieksekusi, karena PyTorch tidak melakukan transfer device secara otomatis dalam sebuah operasi.

### Contoh masalah jika model di GPU dan `X` di CPU

```python
model = model.to("cuda")
X = torch.randn(16, 2)          # tetap di CPU
prediction = model(X)           # akan error
```

Ini akan menghasilkan error seperti:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

Error ini muncul tepat saat forward pass mencoba mengalikan tensor `X` (di CPU) dengan bobot `nn.Linear` (di GPU). Ini konsisten dengan fungsi `inspect_batch()` pada kode soal, yang secara eksplisit mencetak `model device` dan `X.device`/`y.device` — pemeriksaan ini dimaksudkan agar mismatch semacam ini terdeteksi sejak awal sebelum training berjalan lama, bukan tiba-tiba error di tengah training.

### Cara memindahkan tensor atau model ke device yang sesuai

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)        # pindahkan seluruh parameter & buffer model
X = X.to(device)                # pindahkan tensor input
y = y.to(device)                # pindahkan tensor label
```

Catatan penting:
- `model.to(device)` mengubah model **in-place** (memindahkan parameter/buffer-nya), sehingga tidak perlu ditampung ulang, tetapi karena `nn.Module.to()` tetap mengembalikan `self`, penulisan `model = model.to(device)` adalah konvensi yang aman dan umum dipakai.
- `tensor.to(device)` **tidak in-place** — ia mengembalikan tensor baru pada device tujuan, sehingga hasilnya wajib ditampung (`X = X.to(device)`), atau `X` masih tetap berada di device lama.
- Praktik yang baik adalah menentukan `device` sekali di awal skrip, lalu memindahkan model dan setiap batch data ke `device` yang sama secara konsisten sepanjang training maupun evaluasi.

---

# Jawaban — B.9 Analisis Output Layer

> **Soal (bagian "B. Pemeriksaan Input dan Output", No. 9)**
>
> Model yang digunakan adalah:
> ```python
> model = nn.Sequential(
>     nn.Linear(2, 8),
>     nn.ReLU(),
>     nn.Linear(8, 1)
> )
> ```
> 1. Mengapa layer pertama memiliki input size 2?
> 2. Mengapa hidden layer memiliki 8 unit?
> 3. Mengapa output layer memiliki 1 unit?
> 4. Apakah model tersebut menggunakan sigmoid secara eksplisit?
> 5. Mengapa `BCEWithLogitsLoss()` tetap dapat digunakan tanpa menambahkan `nn.Sigmoid()` pada model?

---

## Jawaban

### 1. Mengapa layer pertama memiliki input size 2?

Karena data input `X` memiliki shape `(16, 2)`, artinya setiap sampel direpresentasikan oleh 2 fitur. `nn.Linear(in_features, out_features)` mengharuskan `in_features` sama dengan jumlah fitur pada dimensi terakhir input, sehingga `in_features=2` di sini wajib cocok dengan jumlah kolom `X`. Jika tidak cocok, PyTorch akan melempar error dimensi saat perkalian matriks dilakukan.

### 2. Mengapa hidden layer memiliki 8 unit?

Angka 8 adalah **hyperparameter arsitektur**, dipilih secara bebas oleh perancang model dan bukan ditentukan oleh data. Ini menentukan dimensi ruang representasi tersembunyi (hidden representation) yang menampung fitur hasil kombinasi non-linear dari 2 fitur input. Untuk kasus sederhana seperti pada kode ini (2 fitur, tugas biner), 8 unit cukup memberi kapasitas model untuk mempelajari pola non-linear tanpa membuat model terlalu besar untuk data sekecil ini. Nilai ini bisa diubah dan disetel ulang (tuning) sesuai kompleksitas masalah.

### 3. Mengapa output layer memiliki 1 unit?

Karena tugas yang dikerjakan adalah **klasifikasi biner** (label hanya 0 atau 1). Untuk klasifikasi biner, cukup satu nilai skalar (logit) per sampel yang merepresentasikan skor kecenderungan terhadap kelas positif; probabilitas untuk kelas negatif tinggal `1 - probabilitas_positif`, sehingga tidak diperlukan dua unit output seperti pada klasifikasi multi-kelas.

### 4. Apakah model menggunakan sigmoid secara eksplisit?

**Tidak.** Susunan `nn.Sequential` di atas hanya terdiri dari `Linear → ReLU → Linear`, tidak ada `nn.Sigmoid()` di dalamnya. Artinya output mentah dari model (`prediction`) adalah **logit** — nilai real tak terbatas (bisa negatif atau positif, tidak dibatasi ke rentang 0–1) — bukan probabilitas. Ini bisa dikonfirmasi dari fungsi `evaluate_step()` pada kode program, yang secara eksplisit memanggil `torch.sigmoid(prediction)` secara terpisah setelah forward pass untuk mengubah logit menjadi probabilitas.

### 5. Mengapa `BCEWithLogitsLoss()` tetap bisa dipakai tanpa `nn.Sigmoid()` pada model?

Karena `nn.BCEWithLogitsLoss()` **sudah menggabungkan sigmoid dan binary cross-entropy dalam satu operasi**, sesuai namanya ("with logits" — menerima logit mentah, bukan probabilitas). Secara internal loss ini setara dengan:

```python
loss = nn.BCELoss()(torch.sigmoid(logits), target)
```

tetapi dihitung menggunakan formulasi yang lebih stabil secara numerik (memanfaatkan *log-sum-exp trick*), sehingga menghindari masalah presisi seperti `log(0)` yang dapat muncul jika sigmoid dan BCE dihitung secara terpisah dengan `nn.Sigmoid()` + `nn.BCELoss()`.

Karena alasan ini, menambahkan `nn.Sigmoid()` secara eksplisit di dalam model justru **tidak dianjurkan** ketika memakai `BCEWithLogitsLoss()` — itu akan membuat sigmoid diterapkan dua kali (sekali secara eksplisit di model, sekali lagi secara implisit di dalam loss), yang merusak perhitungan gradient dan loss. Model cukup mengeluarkan logit mentah, dan sigmoid baru diterapkan secara manual saat inference/evaluasi jika probabilitas eksplisit dibutuhkan — persis seperti yang dilakukan pada fungsi `evaluate_step()`.

# Jawaban — C.10 Perbandingan `train()` dan `eval()`

> **Soal (bagian "C. Perbandingan Training dan Evaluation", No. 10)**
>
> Lengkapi tabel berikut:
>
> | Aspek | `model.train()` | `model.eval()` |
> |---|---|---|
> | Tujuan penggunaan | | |
> | Digunakan saat | | |
> | Dropout | | |
> | BatchNorm | | |
> | Gradient tracking | | |
>
> Perhatikan bahwa `model.eval()` tidak sama dengan `torch.no_grad()`. Keduanya mengontrol hal yang berbeda.

---

## Jawaban

| Aspek | `model.train()` | `model.eval()` |
|---|---|---|
| **Tujuan penggunaan** | Mengatur atribut `.training` pada model dan seluruh submodulnya menjadi `True`, sehingga layer yang perilakunya bergantung mode (Dropout, BatchNorm, dll) beroperasi dalam mode training. | Mengatur atribut `.training` pada model dan seluruh submodulnya menjadi `False`, sehingga layer yang sama beroperasi dalam mode evaluasi/inference. |
| **Digunakan saat** | Sebelum training loop dimulai, tepatnya sebelum forward pass pada setiap iterasi training (misal di awal `training_step`). | Sebelum melakukan validasi, testing, atau inference — kapan pun prediksi diambil tanpa tujuan memperbarui bobot. |
| **Dropout** | Aktif: neuron dinonaktifkan secara acak sesuai probabilitas `p`, sisanya diskalakan naik (`1/(1-p)`) agar ekspektasi output tetap konsisten. | Nonaktif: berperilaku sebagai fungsi identitas, seluruh neuron dipertahankan tanpa penskalaan tambahan. |
| **BatchNorm** | Menghitung mean/variance dari batch yang sedang berjalan untuk normalisasi, sekaligus memperbarui `running_mean` dan `running_var` secara akumulatif. | Tidak menghitung statistik dari batch saat ini; memakai `running_mean`/`running_var` yang sudah terkumpul selama training, dan tidak memperbaruinya lagi. |
| **Gradient tracking** | **Tidak dipengaruhi.** `model.train()`/`model.eval()` sama sekali tidak mengontrol apakah gradient dihitung atau tidak — itu adalah tanggung jawab `torch.no_grad()` / `requires_grad`, bukan mode training/eval. | **Tidak dipengaruhi** juga, dengan alasan yang sama. Gradient tetap dapat dihitung dan `.backward()` tetap bisa dipanggil meski model dalam mode `eval()`, kecuali secara eksplisit dibungkus `torch.no_grad()`. |

Baris terakhir sengaja ditulis sama untuk kedua kolom, karena inilah kesalahpahaman paling umum: `model.eval()` sering dikira otomatis mematikan gradient, padahal ia murni saklar perilaku layer, bukan saklar autograd.

---

# Jawaban — C.11 Mengapa Evaluation Menggunakan `torch.no_grad()`?

> **Soal (bagian "C. Perbandingan Training dan Evaluation", No. 11)**
>
> Jelaskan alasan penggunaan berikut saat melakukan validation atau test:
> ```python
> with torch.no_grad():
>     prediction = model(X)
> ```
> Sebutkan minimal dua manfaatnya.

---

## Jawaban

### Alasan penggunaan

Saat validasi atau testing, tujuan forward pass hanyalah **menghasilkan prediksi**, bukan melatih model. Karena itu tidak ada kebutuhan untuk memanggil `loss.backward()` atau `optimizer.step()`, sehingga tidak ada gunanya PyTorch mencatat computation graph untuk operasi ini. `torch.no_grad()` memberi tahu autograd secara eksplisit untuk melewati pencatatan tersebut selama blok kode berjalan, karena hasil forward pass ini memang tidak akan pernah di-backward.

### Manfaat (minimal dua)

1. **Menghemat memori.** Tanpa pencatatan computation graph, PyTorch tidak perlu menyimpan tensor-tensor perantara (intermediate activations) yang biasanya dibutuhkan untuk menghitung gradient saat backward pass. Ini sangat terasa pada model besar atau batch besar, karena aktivasi yang tersimpan bisa memakan memori jauh lebih besar daripada bobot model itu sendiri.

2. **Mempercepat komputasi.** Karena tidak ada overhead pencatatan graph (dan pada beberapa operasi PyTorch bisa memakai kernel yang lebih efisien saat tahu gradient tidak dibutuhkan), forward pass di dalam `torch.no_grad()` berjalan lebih cepat dibanding forward pass biasa dengan autograd aktif.

3. **(Tambahan) Mencegah kesalahan tidak sengaja.** Karena `requires_grad` pada output menjadi `False`, jika ada baris kode yang secara tidak sengaja mencoba memanggil `.backward()` di dalam blok evaluasi, PyTorch akan langsung melempar error — bukan diam-diam menjalankan operasi yang tidak diinginkan.

Catatan: `torch.no_grad()` di sini melengkapi `model.eval()`, bukan menggantikannya (lihat C.10) — keduanya harus dipakai bersamaan agar evaluasi benar dari sisi perilaku layer **dan** efisien dari sisi komputasi gradient.

---

# Jawaban — C.12 Membuat Fungsi Evaluation

> **Soal (bagian "C. Perbandingan Training dan Evaluation", No. 12)**
>
> Buatlah fungsi berikut:
> ```python
> def evaluate_step(model, criterion, X, y):
>     pass
> ```
> Fungsi tersebut harus:
> 1. Mengaktifkan evaluation mode.
> 2. Menonaktifkan gradient tracking.
> 3. Menghasilkan prediksi.
> 4. Menghitung loss.
> 5. Menghitung binary accuracy.
> 6. Mengembalikan nilai loss dan accuracy.

---

## Jawaban

```python
def evaluate_step(model, criterion, X, y):
    model.eval()                          # 1. aktifkan evaluation mode
    with torch.no_grad():                 # 2. nonaktifkan gradient tracking
        prediction = model(X)             # 3. hasilkan prediksi (logit)
        loss = criterion(prediction, y)   # 4. hitung loss

        probabilities = torch.sigmoid(prediction)        # ubah logit -> probabilitas
        predicted_classes = (probabilities >= 0.5).float()
        accuracy = (predicted_classes == y).float().mean()  # 5. hitung binary accuracy

    return loss.item(), accuracy.item()   # 6. kembalikan loss dan accuracy
```

## Penjelasan tiap bagian

- **`model.eval()`** dipanggil di awal agar layer yang bergantung mode (Dropout/BatchNorm) berperilaku sesuai evaluasi, bukan training (lihat C.10).
- **`with torch.no_grad():`** membungkus seluruh proses forward dan perhitungan metrik, karena tidak ada `backward()` yang akan dipanggil saat evaluasi (lihat C.11).
- **`prediction = model(X)`** menghasilkan logit mentah, bukan probabilitas, karena model tidak memakai `nn.Sigmoid()` secara eksplisit (lihat B.9).
- **`loss = criterion(prediction, y)`** memakai `prediction` (logit) langsung, konsisten dengan cara kerja `nn.BCEWithLogitsLoss()` yang menerima logit, bukan probabilitas.
- **`torch.sigmoid(prediction)`** dibutuhkan secara terpisah untuk menghitung accuracy, karena accuracy butuh prediksi kelas (0/1) yang berasal dari probabilitas, bukan dari logit langsung. Ambang batas `0.5` pada probabilitas setara dengan ambang batas `0` pada logit.
- **`(predicted_classes == y).float().mean()`** menghitung proporsi prediksi yang cocok dengan label sebenarnya, yaitu binary accuracy.
- **`.item()`** dipanggil di akhir untuk mengubah tensor skalar menjadi tipe Python murni (`float`), sehingga nilai kembalian lebih mudah dipakai untuk logging (misalnya `print` atau disimpan ke list histori metrik) tanpa membawa beban graph/tensor.

Fungsi ini identik dengan `evaluate_step()` yang sudah ada pada kode program di soal, dan berlaku sebagai kebalikan langsung dari `broken_training_step()`: di mana `broken_training_step()` salah memakai `eval()` untuk training, fungsi ini justru menunjukkan cara `eval()` dan `no_grad()` seharusnya dipakai bersama — khusus untuk fase evaluasi.

# Jawaban — D.13 Membandingkan Fungsi Broken dan Corrected

> **Soal (bagian "D. Eksperimen dan Interpretasi", No. 13)**
>
> Jalankan kedua fungsi berikut:
> - `broken_training_step(...)`
> - `corrected_training_step(...)`
>
> Catat: loss dari fungsi broken, loss dari fungsi corrected, mode model sebelum dan sesudah setiap fungsi, apakah parameter model berubah. Apakah hasil tersebut cukup untuk menyimpulkan bahwa fungsi corrected selalu lebih baik? Jelaskan keterbatasan perbandingan tersebut.

---

## Cara menjalankan

Jalankan skrip aslinya (`DL_Sesi04_Framework_Debugging_Exercise.py`) sesuai blok `__main__`-nya:

```bash
uv run --no-project --with torch DL_Sesi04_Framework_Debugging_Exercise.py
```

Bagian relevan dari `__main__`:

```python
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
X = torch.randn(16, 2)
y = torch.randint(0, 2, (16, 1)).float()

print("broken loss:", broken_training_step(model, optimizer, criterion, X, y))
print("corrected loss:", corrected_training_step(model, optimizer, criterion, X, y))
```

## Hasil yang dicatat

| Pengamatan | `broken_training_step()` | `corrected_training_step()` |
|---|---|---|
| Mode model **sebelum** dipanggil | `model.training = True` (default setelah `nn.Sequential` dibuat) | `model.training = False` (karena `broken_training_step` mengakhiri eksekusinya dalam mode `eval`, lihat baris `model.eval()` di dalamnya) |
| Baris pertama di dalam fungsi | `model.eval()` → `model.training` menjadi `False` | `model.train()` → `model.training` menjadi `True` |
| Mode model **sesudah** dipanggil | `False` (tertinggal dalam mode eval) | `True` (tertinggal dalam mode train) |
| Loss yang dicetak | Sebuah angka (mis. sekitar 0.6–0.8, tepatnya tergantung seed/mesin) | Angka lain, biasanya sedikit lebih rendah dari loss broken |
| Apakah parameter model berubah? | **Ya.** `optimizer.step()` tetap dipanggil dan tetap memperbarui bobot, karena `model.eval()` tidak menghalangi `backward()`/`step()` (lihat A.1 dan A.3) | **Ya**, bobot berubah lagi untuk kedua kalinya, melanjutkan dari bobot hasil `broken_training_step()` |

*(Angka loss persis bergantung pada seed, versi PyTorch, dan device; yang penting dicatat di sini adalah **pola**-nya, bukan nilai eksaknya. Isikan nilai loss aktual dari output terminal Anda ke tabel ini.)*

## Apakah hasil ini cukup untuk menyimpulkan `corrected` selalu lebih baik?

**Tidak cukup.** Perbandingan loss ini tidak fair dan tidak bisa dijadikan bukti bahwa versi `corrected` "selalu lebih baik", karena beberapa alasan berikut.

## Keterbatasan perbandingan

1. **Kedua fungsi dijalankan berurutan pada model dan optimizer yang sama.** `corrected_training_step()` dipanggil **setelah** `broken_training_step()` sudah mengubah bobot model satu langkah lebih dulu. Artinya loss dari `corrected_training_step()` dihitung dari titik awal bobot yang *berbeda* (sudah ter-update sekali), bukan dari kondisi awal yang identik dengan `broken_training_step()`. Selisih loss yang teramati sebagian bisa disebabkan oleh fakta bahwa ini adalah langkah optimasi ke-2, bukan murni karena mode `train`/`eval`.

2. **Optimizer (Adam) memiliki state internal yang juga ikut ter-update.** Adam menyimpan `running average` dari gradient (`m`) dan gradient kuadrat (`v`) untuk tiap parameter. Setelah `optimizer.step()` dipanggil sekali di `broken_training_step()`, state ini sudah tidak nol lagi saat `corrected_training_step()` dipanggil, sehingga arah dan besar update kedua tidak murni mencerminkan gradient batch saat itu saja.

3. **Hanya satu batch data dan satu langkah update.** Dengan model sekecil ini (`Linear-ReLU-Linear`, tanpa Dropout/BatchNorm), Eksperimen 2 pada bagian A.2 sudah menunjukkan bahwa secara numerik **tidak ada perbedaan** antara training di mode `eval` vs `train` — karena tidak ada layer yang perilakunya berubah antar mode. Jadi kalau ada selisih loss yang teramati dalam satu langkah ini, itu murni artefak dari poin 1 dan 2 (urutan eksekusi dan state optimizer), **bukan** bukti bahwa mode `eval`/`train` menyebabkan perbedaan performa pada model ini.

4. **Satu langkah training tidak merepresentasikan performa jangka panjang.** Loss dari satu forward-backward-step tidak bisa dipakai untuk menyimpulkan kualitas training secara umum. Kesimpulan yang valid membutuhkan training berulang (banyak epoch), idealnya dengan train/validation split, dan idealnya diulang dengan beberapa seed berbeda untuk melihat rata-rata dan variansinya (seperti pendekatan pada Eksperimen 5 sebelumnya, yang melatih penuh selama 300 epoch dan merata-rata 5 seed).

5. **Untuk perbandingan yang adil**, seharusnya kedua fungsi dijalankan dari **kondisi awal yang identik** — model dengan bobot awal yang sama (`copy.deepcopy` atau `torch.manual_seed` ulang sebelum inisialisasi), dan optimizer yang baru/fresh untuk masing-masing, sebagaimana dilakukan pada Eksperimen 2 dan Eksperimen 5 di bagian sebelumnya. Perbedaan mode `train`/`eval` baru benar-benar terlihat dampaknya jika model memiliki Dropout atau BatchNorm (lihat Eksperimen 3 dan 4).

---

# Jawaban — D.14 Mengapa Model Kecil Dapat Tetap Memiliki Masalah?

> **Soal (bagian "D. Eksperimen dan Interpretasi", No. 14)**
>
> Model yang digunakan sangat kecil (`nn.Linear(2, 8)`). Apakah model kecil pasti bebas dari bug? Jelaskan minimal tiga masalah yang tetap dapat terjadi.

---

## Jawaban

**Tidak.** Ukuran model (jumlah layer atau parameter) sama sekali tidak berkaitan dengan benar atau salahnya *training loop* di sekitarnya. Bug-bug yang dibahas sepanjang latihan ini (bagian A–C) bersifat **struktural/mekanis**, bukan bergantung pada kompleksitas arsitektur. Berikut minimal tiga (dipilih lima) masalah yang tetap bisa terjadi meski model hanya `Linear(2,8) → ReLU → Linear(8,1)`:

### 1. Shape input tidak sesuai

Jika `X` diberikan dengan shape `(16, 3)` alih-alih `(16, 2)` — misalnya karena ada satu kolom fitur ekstra yang lupa dibuang saat preprocessing — `nn.Linear(2, 8)` akan langsung melempar `RuntimeError` karena dimensi terakhir input tidak cocok dengan `in_features`. Model sekecil apa pun tetap bergantung sepenuhnya pada kecocokan shape data dengan definisi layer pertamanya (lihat B.6).

### 2. Label memiliki dtype yang salah

Jika baris `.float()` pada `y = torch.randint(0, 2, (16, 1)).float()` terlewat, `y` akan tetap bertipe `int64`. `nn.BCEWithLogitsLoss()` akan gagal atau memberi hasil yang tidak diharapkan karena target diharapkan floating-point (lihat B.7). Ini murni masalah tipe data, tidak ada hubungannya dengan ukuran model.

### 3. Model dan data berada pada device yang berbeda

Jika suatu saat model dipindahkan ke GPU (`model.to("cuda")`) tetapi `X`/`y` lupa dipindahkan, akan muncul `RuntimeError: Expected all tensors to be on the same device...` (lihat B.8) — terlepas dari apakah model hanya dua layer atau seratus layer.

### 4. Loss function tidak cocok dengan output model

Jika model tetap mengeluarkan 1 logit (`Linear(8, 1)`) tapi memakai `nn.CrossEntropyLoss()` (yang mengharapkan output berdimensi `[batch, jumlah_kelas]` dengan `jumlah_kelas ≥ 2`, dan label `int64` berisi indeks kelas, bukan target float), kode akan error atau menghasilkan loss yang secara konsep tidak masuk akal.

### 5. Mode training dan evaluation tertukar

Inilah bug utama soal ini: memanggil `model.eval()` alih-alih `model.train()` saat training (lihat A.1–A.3). Pada model sekecil ini dampaknya memang tidak terlihat secara numerik karena tidak ada Dropout/BatchNorm, tetapi begitu model ini "naik kelas" — misalnya ditambah satu baris `nn.Dropout(0.3)` — bug yang sama langsung berdampak nyata pada hasil training (lihat Eksperimen 3 & 5). Bug-nya sendiri sudah ada sejak awal, hanya belum "terpicu" oleh arsitektur yang sesederhana ini.

### 6. Gradient tidak dihapus (`zero_grad()` terlewat)

Terlepas dari ukuran model, jika `optimizer.zero_grad()` hilang, gradient akan terus terakumulasi antar-batch (lihat A.5), membuat update bobot membesar tak terkendali dan berisiko membuat loss meledak — ini murni soal manajemen state optimizer, bukan soal kapasitas model.

### 7. Learning rate tidak sesuai

Learning rate yang terlalu besar (misalnya `lr=1.0` alih-alih `1e-3`) bisa membuat loss berosilasi atau divergen, sementara learning rate yang terlalu kecil membuat training berjalan sangat lambat — keduanya berlaku sama untuk model dua-layer maupun model besar.

## Kesimpulan

Model kecil memang mengurangi risiko masalah yang berkaitan dengan **kapasitas/kompleksitas** (misalnya overfitting parah atau vanishing gradient pada jaringan sangat dalam), tetapi sama sekali tidak melindungi dari bug-bug **mekanis** di seputar training loop — shape, dtype, device, kecocokan loss function, mode `train`/`eval`, manajemen gradient, dan hyperparameter. Justru karena dampaknya sering tidak terlihat pada model kecil (seperti kasus `eval()` di soal ini), bug semacam ini lebih berbahaya: ia lolos tanpa terdeteksi sampai model diperbesar atau arsitekturnya diubah, di titik mana efeknya baru muncul dan sulit ditelusuri sumbernya.

---

# Jawaban — D.15 Membuat Checklist Debugging

> **Soal (bagian "D. Eksperimen dan Interpretasi", No. 15)**
>
> Buat checklist minimal delapan pemeriksaan yang harus dilakukan sebelum menjalankan training panjang. Tambahkan minimal empat pemeriksaan lain di luar contoh yang diberikan.

---

## Jawaban

### Checklist dari contoh soal

- [ ] Shape input (`X`) sudah diperiksa dan sesuai dengan `in_features` layer pertama
- [ ] Shape label (`y`) sudah diperiksa dan sesuai dengan shape output model (hindari broadcasting tak sengaja, lihat B.6)
- [ ] Dtype input dan label sudah sesuai dengan yang dibutuhkan loss function (mis. `float32` untuk `BCEWithLogitsLoss`, lihat B.7)
- [ ] Model dan seluruh tensor data berada pada device yang sama (`cpu`/`cuda`), dicek lewat `next(model.parameters()).device` vs `X.device` (lihat B.8)

### Pemeriksaan tambahan

- [ ] `model.train()` dipanggil sebelum training loop, dan `model.eval()` + `torch.no_grad()` dipanggil sebelum validasi/testing — bukan tertukar (lihat A.1–A.3, C.10)
- [ ] `optimizer.zero_grad()` dipanggil di setiap iterasi sebelum `backward()`, agar gradient tidak terakumulasi lintas batch (lihat A.5)
- [ ] Loss function cocok dengan output model dan tugasnya: logit mentah tanpa `nn.Sigmoid()` untuk `BCEWithLogitsLoss`, tidak dobel sigmoid, target berformat sesuai (lihat B.9)
- [ ] Urutan `zero_grad() → forward → loss → backward() → step()` sudah benar dan tidak tertukar (lihat A.4)
- [ ] Loss pada satu/dua batch pertama turun secara wajar (sanity check overfit pada sample kecil) sebelum menjalankan training penuh berjam-jam
- [ ] Learning rate dan optimizer sudah masuk akal untuk skala data/model (tidak terlalu besar hingga loss `NaN`/meledak, tidak terlalu kecil hingga stagnan)
- [ ] Ada pemisahan data train/validation yang jelas, dan data validation tidak pernah dipakai untuk `backward()`/update bobot
- [ ] Reproducibility: `torch.manual_seed(...)` (dan seed library lain yang relevan, misal `numpy`/`random`) sudah diset agar eksperimen bisa diulang dan dibandingkan secara adil
- [ ] Checkpoint/logging sudah disiapkan (menyimpan bobot & metrik secara berkala) sehingga training panjang tidak hilang begitu saja jika proses terhenti di tengah jalan
- [ ] Batch terakhir/tidak penuh (*drop_last*) dan ukuran batch pada `DataLoader` sudah diperiksa, terutama jika memakai layer seperti `BatchNorm` yang sensitif terhadap ukuran batch sangat kecil (batch size = 1 bisa membuat `BatchNorm` error atau tidak stabil)

Checklist ini mencerminkan kembali seluruh bug yang dibahas di bagian A–C soal ini: setiap poin di atas berkorespondensi langsung dengan satu jenis kegagalan yang sudah dianalisis dan dibuktikan lewat eksperimen sebelumnya (mode train/eval, gradient accumulation, shape, dtype, device, dan kecocokan loss function).

# Jawaban — E.16 Penjelasan Teknis

> **Soal (bagian "E. Pertanyaan Reflektif", No. 16)**
>
> Jelaskan mengapa kode yang tidak menghasilkan error belum tentu merupakan kode yang benar? Berikan satu contoh dari latihan ini.

---

## Jawaban

Kode yang bebas error hanya membuktikan bahwa kode tersebut **valid secara sintaksis dan semantik pada level Python/PyTorch** — setiap pemanggilan fungsi memiliki tipe data dan shape yang bisa diterima oleh operasi berikutnya, sehingga interpreter/runtime tidak menemukan alasan untuk berhenti. Namun tidak adanya error **tidak** membuktikan bahwa kode tersebut melakukan apa yang **dimaksudkan secara logis/matematis** oleh pembuatnya. PyTorch (dan bahasa pemrograman pada umumnya) hanya bisa mendeteksi kesalahan yang melanggar aturan tipe, shape, atau device — bukan kesalahan berupa "operasi yang secara teknis sah tapi secara konsep keliru".

### Contoh dari latihan ini

Contoh paling jelas adalah `broken_training_step()` pada bagian A. Fungsi ini memanggil `model.eval()` alih-alih `model.train()` sebelum forward pass saat training. Kode ini:

- **Tidak menghasilkan error apa pun** — `model(X)`, `loss.backward()`, dan `optimizer.step()` semuanya berjalan mulus, karena `model.eval()` secara sintaksis maupun semantik adalah pemanggilan method yang sah (lihat A.1, A.3).
- **Tetapi secara konsep salah** — Dropout tidak pernah aktif, BatchNorm tidak pernah mempelajari statistik data (dibuktikan lewat Eksperimen 3 dan 4), sehingga model tidak dilatih sebagaimana mestinya. Bug ini bahkan bisa lolos tanpa terdeteksi sama sekali jika arsitektur modelnya tidak memiliki Dropout/BatchNorm, seperti terbukti pada Eksperimen 2 (bobot akhir mode `eval` dan `train` identik untuk model `Linear-ReLU-Linear`).

Ini menunjukkan bahwa keberhasilan eksekusi (`no error`) dan kebenaran logis (`correctness`) adalah dua hal yang sepenuhnya berbeda, dan pengujian kode deep learning tidak cukup hanya dengan memastikan skrip berjalan sampai akhir — perlu verifikasi tambahan seperti memeriksa nilai/statistik intermediate, membandingkan dengan ekspektasi teoretis, atau menjalankan eksperimen kontrol seperti yang dilakukan sepanjang latihan ini.

---

# Jawaban — E.17 Peran Framework

> **Soal (bagian "E. Pertanyaan Reflektif", No. 17)**
>
> PyTorch menyediakan automatic differentiation dan training utilities. Mengapa mahasiswa tetap perlu memahami hal-hal berikut: forward pass, loss function, backward pass, optimizer update, training mode, evaluation mode, shape dan dtype?

---

## Jawaban

PyTorch memang mengotomasi banyak hal — terutama perhitungan gradient lewat *autograd* — tetapi otomasi ini hanya menangani **bagaimana** sebuah operasi dihitung secara efisien, bukan **apakah** operasi tersebut benar untuk masalah yang sedang dikerjakan. Framework tidak bisa membaca maksud pengguna; ia hanya menjalankan apa yang diperintahkan. Karena itu, pemahaman mendalam terhadap setiap komponen berikut tetap krusial:

- **Forward pass** — menentukan bagaimana data mengalir dan diubah menjadi prediksi; kesalahan arsitektur (jumlah layer, urutan, fungsi aktivasi) tidak akan terdeteksi otomatis oleh PyTorch selama shape-nya kompatibel (lihat B.9).
- **Loss function** — memilih loss yang salah untuk suatu tugas (misalnya `CrossEntropyLoss` untuk kasus biner yang seharusnya `BCEWithLogitsLoss`) bisa saja tetap berjalan tanpa error jika kebetulan dimensinya cocok, tetapi menghasilkan sinyal training yang secara matematis tidak sesuai (lihat B.7, B.9).
- **Backward pass** — autograd menghitung gradient secara otomatis, tetapi mahasiswa perlu tahu *apa* yang sedang di-backward-kan (loss skalar, bukan prediksi) dan *mengapa* urutan operasi penting (lihat A.4), karena kesalahan urutan tidak selalu memicu error.
- **Optimizer update** — memahami bahwa gradient bersifat akumulatif dan perlu direset (`zero_grad()`) adalah pengetahuan tentang *state* optimizer yang tidak diberitahukan oleh framework secara eksplisit; melupakannya tidak menghasilkan error, hanya training yang perlahan rusak (lihat A.5).
- **Training mode dan evaluation mode** — `model.train()`/`model.eval()` adalah saklar perilaku yang harus disadari dan dikelola manual oleh pengguna di titik yang tepat; PyTorch tidak akan memperingatkan jika mode yang dipakai keliru, karena secara teknis kedua mode sama-sama valid untuk dijalankan (lihat A.1–A.3, C.10).
- **Shape dan dtype** — inilah "kontrak" antar-operasi yang harus dijaga manual; PyTorch memang akan error jika benar-benar tidak kompatibel, tetapi pada kasus seperti broadcasting (label `(16,)` vs output `(16,1)`) justru **tidak** error dan diam-diam menghasilkan perhitungan yang salah (lihat B.6).

Singkatnya, automatic differentiation mengotomasi **komputasi**, bukan **penalaran**. Mahasiswa yang hanya mengandalkan framework tanpa memahami mekanisme di baliknya akan kesulitan mendiagnosis kegagalan yang bersifat diam-diam (silent bugs) — yaitu justru jenis bug yang paling sering muncul dan paling sulit dilacak dalam praktik deep learning sehari-hari, sebagaimana ditunjukkan berulang kali sepanjang latihan ini.

---

# Jawaban — E.18 Kesimpulan

> **Soal (bagian "E. Pertanyaan Reflektif", No. 18)**
>
> Tuliskan kesimpulan sepanjang 150–250 kata yang menjelaskan: (1) masalah utama pada `broken_training_step()`, (2) perbaikan pada `corrected_training_step()`, (3) perbedaan `model.train()`, `model.eval()`, dan `torch.no_grad()`, (4) pentingnya pemeriksaan shape, dtype, device, dan gradient, (5) satu pelajaran penting dari latihan ini.

---

## Jawaban

Latihan ini bermula dari satu bug tersembunyi pada `broken_training_step()`: fungsi tersebut memanggil `model.eval()` alih-alih `model.train()` sebelum forward pass, padahal fungsinya justru dipakai untuk melatih model. Masalah utamanya adalah `eval()` tidak menghentikan `backward()` atau `optimizer.step()`, sehingga bobot tetap diperbarui, tetapi layer seperti Dropout dan BatchNorm ikut berperilaku seolah sedang evaluasi, membuat regularisasi mati dan statistik BatchNorm tidak pernah dipelajari. Perbaikannya sederhana, mengganti satu baris menjadi `model.train()`, namun perbaikan ini krusial karena memulihkan perilaku layer yang benar selama training. Dari sini menjadi jelas perbedaan tiga mekanisme yang sering tertukar: `model.train()`/`model.eval()` mengatur perilaku layer, sementara `torch.no_grad()` mengatur pencatatan gradient; ketiganya independen dan evaluasi yang benar membutuhkan `model.eval()` bersama `torch.no_grad()`. Latihan ini juga menegaskan pentingnya memeriksa shape, dtype, device, dan status gradient sebelum training panjang, karena kesalahan pada aspek-aspek ini sering tidak memicu error, hanya membuat hasil training salah secara diam-diam. Pelajaran terpenting yang saya peroleh adalah bahwa kode yang berjalan tanpa error bukan jaminan kode itu benar; debugging deep learning menuntut verifikasi eksplisit terhadap asumsi, bukan sekadar mengandalkan tidak adanya exception.
