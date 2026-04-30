# UTS Pengantar Deep Learning
Tiga kasus klasifikasi dengan pendekatan classical ML vs deep learning.

## Dependencies
torch, scikit-learn, pandas, numpy, matplotlib, seaborn, pytorch-tabnet, scikit-image

## Kasus
**Kasus 1 — Titanic** (`case_1_titanic.ipynb`)
Prediksi penumpang selamat atau tidak. Binary classification pada data tabular.
Classical: Logistic Regression + Random Forest | DL: MLP + TabNet

**Kasus 2 — MNIST Digit Recognizer** (`case_2_mnist.ipynb`)
Klasifikasi digit tulisan tangan 0-9. Multi-class image classification.
Classical: HOG + LinearSVM, PCA + KNN | DL: CNN

**Kasus 3 — Disaster Tweets** (`case_3_tweets.ipynb`)
Deteksi apakah tweet membicarakan bencana nyata atau bukan. Binary NLP classification.
Classical: TF-IDF + Logistic Regression, TF-IDF + LinearSVM | DL: Embedding + BiLSTM

## Struktur Folder

```
uts_deeplearning/
├── case_1_titanic.ipynb
├── case_2_mnist.ipynb
├── case_3_tweets.ipynb
├── titanic_dataset/
├── mnist_dataset/
├── tweet_dataset/
├── submission_titanic.csv
├── submission_mnist.csv
└── submission_tweets.csv
```

## Cara Jalankan

Aktifkan virtual environment dulu:

```
venv_dl\Scripts\activate   # Windows
```

Buka notebook di VS Code / Jupyter, lalu Restart & Run All.

