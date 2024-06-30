# Twitter Sentiment Preprocessing
Project ini mendemonstrasikan proses preprocessing data twitter untuk keperluan sentiment analysis

## Dataset Overview
Dataset Sentiment 140 memuat 1,6 juta tweet yang diekstraksi menggunakan twitter API. Data dianotasi dengan label 0 yang mengindikasikan negatif dan 4 yang mengindikasikan positif. Dataset ini terdiri dari 6 kolom :
- target : Sentiment
- ids : Tweet ID
- date : Tanggal tweet
- flag : Query Flag
- user : Username penulis tweet
- text : Teks tweet

Sumber : Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12. [(Download Paper)](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

Download Dataset : [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)


## Text Preprocessing

Langkah awal dalam project ini adalah melakukan teks preprocessing dari data yang ada. Setelah dataset dimuat dilakukan beberapa penyesuaian antara lain:
1. Hanya memilih kolom yang relevan yaitu kolom `text` sebagai data tweet yang akan dianalisa dan kolom `target` yang merupakan sentiment untuk diprediksi.
2. Mengganti label untuk sentiment positif dari 4 menjadi 1, sehingga hasil akhir label yang akan diprediksi adalah 0 untuk negatif dan 1 untuk positif.

Selanjutnya adalah melakukan data cleaning untuk memastikan data yang diproses oleh machine learning adalah data yang bersih dan tidak banyak noise. Berikut tahapan preprocessing yang dilakukan.
1. **Menghapus URL**
2. **Menghapus Username**
3. **Menghapus Hashtag**
4. **Menghapus Tanda Baca**
5. **Konversi ke Huruf Kecil**
6. **Tokenisasi** : Memecah teks menjadi token kata-kata individual.
7. **Menghapus Stopwords** : Menghapus kata-kata umum seperti kata sambung (misalnya, "the", "is", "in").
8. **Stemming**: Mengubah kata ke dalam bentuk dasarnya.

## Text Vectorizing
Pada tahap ini dilakukan proses mengubah teks menjadi representasi numerik. Bentuk data yang numerik akan memudahkan machine learning untuk melakukan pemrosesan model. Untuk tahap ini, digunakan metode TF-IDF yang biasa diterapkan untuk menilai seberapa penting sebuah kata dalam sebuah dokumen relatif terhadap sekumpulan dokumen (corpus). TF-IDF merupakan hasil perkalian Term Frequency (TF) dan Inverse Document Frequency (IDF).

**Term Frequency (TF)**

Term Frequency (TF) mengukur seberapa sering sebuah kata muncul dalam sebuah dokumen. 

$$ \text{TF}(t, d) = \frac{\text{Jumlah kemunculan kata } t \text{ dalam dokumen } d}{\text{Jumlah total kata dalam dokumen } d} $$

**Inverse Document Frequency (IDF)**

Inverse Document Frequency (IDF) mengukur seberapa penting sebuah kata di dalam seluruh corpus. 

$$ \text{IDF}(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right) $$

Di mana:
- \( N \) adalah jumlah total dokumen dalam corpus.
- $(|\{d \in D : t \in d\}|)$ adalah jumlah dokumen yang mengandung kata \( t \).

**TF-IDF**

TF-IDF adalah hasil perkalian dari TF dan IDF.

$$ \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D) $$

Di mana:
- \( t \) adalah kata tertentu.
- \( d \) adalah dokumen tertentu dalam corpus.
- \( D \) adalah seluruh corpus dokumen.


## Text Splitting
Untuk memungkinkan evaluasi model secara lebih objektif. Dilakukan teks splitting dengan membagi data keseluruhan menjadi 80% untuk training dan 20% testing. Adanya data testing yang tidak digunakan untuk pelatihan berguna untuk memberikan gambaran akurasi prediksi model yang belum pernah dilihat sebelumnya.
