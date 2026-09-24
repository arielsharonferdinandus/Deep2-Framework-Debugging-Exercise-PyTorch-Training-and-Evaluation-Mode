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
