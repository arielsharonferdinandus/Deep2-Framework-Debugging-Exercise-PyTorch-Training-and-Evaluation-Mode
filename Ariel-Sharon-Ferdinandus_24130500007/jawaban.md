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
