## 📘 RAG Parameter Exploration Assignment

Pada tugas ini kamu akan bereksperimen dengan **5 parameter utama Retrieval-Augmented Generation (RAG)** + **opsi model Groq**.

Kamu akan:

✅ memahami efek tuning RAG

✅ mengamati perubahan kualitas vs kecepatan

✅ belajar trade-off retrieval & generation

✅ menulis insight praktis dari eksperimen

> Tools sudah disediakan. Kamu **tidak perlu ngoding**, hanya ubah nilai parameter di file Python & restart server.

**Notes:**

**- Ikuti video di LMS jika kamu kebingungan.**

**- Sediakan penyimpanan at least 2GB, bakalan ada download model yang cukup besar ke laptopmu.**

**- Hapus proyek ini kalau sudah selesai, biar laptop mu tidak tercekik**

**- Ketika proyek ini sudah diisi, sudah dikerjakan dibawah, maka upload ke GitHub.**

---

## ⚙️ Parameter yang diuji

### 1️⃣ Ukuran chunk

```python
CHUNK_SIZE = 300
```

**Fungsi**
Mengatur seberapa besar potongan teks untuk embedding.

| Arah tuning               | Efek                                                                 |
| ------------------------- | -------------------------------------------------------------------- |
| **Lebih besar** (400–600) | jawabannya langsung to the point tentang AR dan alasan pemilihannya. |
| **Lebih kecil** (150–250) | jawabannya melebar ke banyak hal lain yang sebenarnya tidak ditanya. |

**Pertanyaan**

- Chunk besar atau kecil lebih tepat untuk dokumen yang kamu tes?
- Jawab : menggunakan chunk besar lebih cocok.

---

### 2️⃣ Kandidat awal vector search

```python
TOP_K_INITIAL = 20
```

**Fungsi**
Berapa dokumen paling mirip yang diambil sebelum filtering.

| Arah tuning         | Efek                                                                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Naikkan** (30–50) | Sistem mengambil lebih banyak kandidat dokumen sebelum memilih yang terbaik. Hasilnya lebih aman untuk pertanyaan yang agak luas, tapi prosesnya sedikit lebih lama. |
| **Turunkan** (5–10) | Sistem hanya mengambil sedikit kandidat dokumen. Lebih cepat, tapi berisiko kehilangan konteks penting jika dokumen yang cocok tidak termasuk dalam 5–10 pertama.    |

**Pertanyaan**

- Apakah top-k kecil cukup untuk dataset kamu?
- Jawab : Ya, top-k kecil sudah cukup.

---

### 3️⃣ Konten final ke LLM

```python
FINAL_TOP_M = 5
```

**Fungsi**
Jumlah konteks yang akhirnya dimasukkan ke prompt LLM.

| Arah tuning        | Efek                                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Naikkan** (6–8)  | LLM menerima lebih banyak potongan informasi. Jawaban jadi lebih lengkap dan aman dari “konteks hilang”, tapi kadang membuat respon lebih panjang atau memasukkan info yang kurang relevan. |
| **Turunkan** (2–3) | LLM hanya memakai sedikit konteks. Jawaban jadi lebih ringkas dan langsung ke poin, tapi ada risiko bagian penting tidak ikut terbawa.                                                      |

**Pertanyaan**

- Benarkah “semakin banyak konteks semakin bagus”?
- Jawab : Lebih banyak konteks bisa membantu, tapi tidak berarti selalu lebih baik. Kalau konteks terlalu banyak, jawabannya bisa jadi melebar.

---

### 4️⃣ Cross-Encoder Re-ranker

```python
USE_RERANKER = True
```

**Fungsi**
Menilai ulang (rerank) dokumen hasil embedding.

| Mode    | Efek                                                                                                                                                                                           |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `True`  | Sistem mengecek ulang, lalu memilih mana yang paling tepat sebelum menjawab. Hasilnya biasanya lebih akurat untuk pertanyaan yang rumit atau membingungkan, tapi waktu proses jadi lebih lama. |
| `False` | Sistem langsung memakai hasil tanpa dicek ulang. Prosesnya lebih cepat, tapi bisa saja mengambil dokumen yang kurang tepat untuk pertanyaan tertentu.                                          |

**Pertanyaan**

- Pada query jenis apa reranker paling terasa manfaatnya?
- Jawab : Reranker paling terasa manfaatnya pada pertanyaan yang lebih sulit, Contohnya seperti pertanyaan kamu: “Mengapa Pertamina RU III mengurusi ikan Belida?”. Ini bukan pertanyaan sederhana. Jawabannya tersebar di beberapa bagian dokumen, dan membutuhkan pemahaman konteks (CSR, edukasi, pelestarian, studi kasus, dan alasan pemilihan lokasi).

---

### 5️⃣ Temperatur generasi

```python
TEMPERATURE = 0.2
```

**Fungsi**
Mengontrol variasi / kreativitas jawaban.

| Value   | Efek                                                                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0.0–0.2 | Jawaban tidak bertele-tele, langsung ke poin, dan mengikuti dokumen. Cocok untuk jawaban faktual atau formal.                                    |
| 0.5–0.7 | Jawaban lebih bervariasi, bahasanya lebih bebas tetapi bisa sedikit melebar dari dokumen. Cocok untuk penjelasan yang lebih santai atau kreatif. |

**Pertanyaan**

- Untuk QA berbasis dokumen perusahaan, temperatur ideal berapa?
- Jawab : 0.0–0.2 alasannya karena QA berbasis dokumen perusahaan harus konsisten, tidak boleh menambah informasi yang tidak ada di dokumen, dan harus menjaga “gaya formal” perusahaan.

---

### 6️⃣ Model LLM Groq

```python
GROQ_MODEL = "llama-3.1-8b-instant"
# alternatif:
# gpt-oss-120b
# llama-3.3-70b-versatile
```

| Pilihan                  | Efek                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| Model kecil (8B)         | Lebih cepat, ringan, dan responsnya stabil namun pemahaman konteksnya tidak sedalam model besar.          |
| Model besar (70B / 120B) | Jawaban lebih detail dan lebih “pintar” dalam memahami konteks yang rumit. Namun waktu proses lebih lama. |

**Pertanyaan**

- Apakah peningkatan kualitas sebanding dengan penambahan waktu?
- Jawab : Tidak selalu. Model besar hanya terasa lebih berguna pada pertanyaan yang benar-benar rumit atau membutuhkan penalaran yang tinggi.

---

## Tips

- Pakai PDF minimal 1 halaman, maksimal 10 halaman, biar gak berat.
- Coba pertanyaan definisi, konteks, dan reasoning
- Catat kapan sistem salah & mengapa
- Prioritaskan kualitas hasil dalam laporan

---
