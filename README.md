# 🍌 AIO2026 Banana — Vietnamese Toxic Comment Classification

> **Dự án phân loại bình luận độc hại tiếng Việt** trong khuôn khổ chương trình **AIO 2026**.  
> Xây dựng baseline model và tối ưu hóa bằng pipeline TF-IDF + Machine Learning truyền thống.

---

## 📌 Mục tiêu

Xây dựng và tối ưu mô hình phân loại nhị phân nhằm nhận diện bình luận **độc hại (toxic)** trong văn bản tiếng Việt:

| Nhãn | Ý nghĩa | Mã số |
|------|---------|-------|
| `POS` | Bình thường (non-toxic) | `0` |
| `NEG` | Độc hại (toxic) | `1` |

---

## 📁 Cấu trúc dự án

```
AIO2026-Banana/
├── Baseline_model_and_optimize.ipynb   # Notebook chính: EDA + Preprocessing + Baseline + Optimize
├── notebooks/
│   └── AIE_data.ipynb                  # Notebook phân tích dữ liệu bổ sung
├── README.md
└── .gitignore
```

> **Output (sau khi chạy notebook):**
> ```
> train_clean.csv          # Tập train đã xử lý
> val_clean.csv            # Tập validation
> test_clean.csv           # Tập test
> tfidf_vectorizer.pkl     # TF-IDF vectorizer đã fit
> toxic_ensemble_model.pkl # Mô hình ensemble cuối cùng
> ```

---

## 📊 Dữ liệu

- **File**: `train-00000-of-00001.parquet`
- **Tổng mẫu**: 184,349 bình luận tiếng Việt
- **Cột dữ liệu**:
  - `vi_context` → văn bản tiếng Việt (đã word segmentation, dùng `_` nối từ ghép)
  - `label` → `POS` (non-toxic) / `NEG` (toxic)
- **Phân bố nhãn**: cân bằng (~50/50)

```
label
NEG    92,271  (50.05%)
POS    92,078  (49.95%)
```

---

## 🧹 Pipeline Xử lý

### 1. Tiền xử lý văn bản (Vietnamese Text Preprocessing)

```python
def preprocess_text(text):
    text = html.unescape(text)                       # Giải mã HTML entities
    text = normalize_unicode(text)                   # Chuẩn hóa Unicode NFC
    text = text.lower()                              # Chuyển về chữ thường
    text = normalize_repeated_chars(text)            # "nguuuu" → "ngu"
    text = remove_punctuation_keep_underscore(text)  # Giữ "_" (word seg)
    text = clean_whitespace(text)                    # Xóa khoảng trắng thừa
    text = normalize_repeated_words(text)            # "ngu ngu ngu" → "ngu"
    return text
```

> 💡 **Lý do giữ `_`**: Dataset đã được word segmentation, ví dụ `quan_hệ_tình_dục`, `có_thể`. Giữ `_` giúp TF-IDF coi cụm từ đó là **1 token có ngữ nghĩa**.

### 2. Feature Engineering

| Feature | Mô tả |
|---------|-------|
| `char_len` | Số ký tự trong comment đã clean |
| `word_len` | Số từ trong comment đã clean |

### 3. Lọc & Làm sạch dữ liệu

- Xóa missing values
- Loại comment quá ngắn (`char_len < 2`, `word_len < 1`)
- Xóa dòng trùng lặp (`clean_text` + `label`)
- Xử lý nhãn mâu thuẫn (cùng text, khác label)

### 4. Chia tập dữ liệu

```
Train  : 80%  (stratified)
Val    : 10%  (stratified)
Test   : 10%  (stratified)
```

---

## 🤖 Mô hình

### 📍 Baseline Model
- **Logistic Regression** với TF-IDF (unigram + bigram, `max_features=10,000`)
- Đánh giá nhanh trên tập validation làm điểm tham chiếu ban đầu

### 🔧 Optimize — GridSearchCV

Tinh chỉnh hyperparameter cho 3 mô hình:

| Mô hình | Tham số tìm kiếm |
|---------|-----------------|
| Logistic Regression | `C` ∈ {0.001, 0.01, 0.1, 1, 10} |
| LinearSVC | `C` ∈ {0.001, 0.005, 0.1, 1, 10, 20} |
| LightGBM | `n_estimators`, `learning_rate`, `max_depth` |

### 🏆 Ensemble (Voting Classifier)

Kết hợp 4 mô hình đã tối ưu để ra quyết định cuối:

```python
VotingClassifier(
    estimators=[
        ('lr',   best_lr),        # Logistic Regression (tuned)
        ('svc',  best_svc),       # LinearSVC (tuned)
        ('lgbm', best_lgbm),      # LightGBM (tuned)
        ('nb',   MultinomialNB),  # Naive Bayes
    ],
    voting='hard'
)
```

---

## ⚙️ Cài đặt & Chạy

### Yêu cầu

```bash
pip install pandas scikit-learn lightgbm joblib
```

### Chạy notebook

1. Upload `train-00000-of-00001.parquet` lên Google Colab
2. Mở `Baseline_model_and_optimize.ipynb`
3. Chạy toàn bộ cells theo thứ tự từ trên xuống

### Dùng mô hình đã lưu

```python
import joblib

vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('toxic_ensemble_model.pkl')

text = ["Mày ngu thật sự"]
X = vectorizer.transform(text)
pred = model.predict(X)
print("Toxic" if pred[0] == 1 else "Non-toxic")
```

---

## 📈 Kết quả

| Mô hình | Tập đánh giá |
|---------|-------------|
| Logistic Regression (baseline) | Validation set |
| Ensemble (LR + SVC + LightGBM + NB) | Validation + Test set |

> Chi tiết Precision, Recall, F1-score xem trong **`Baseline_model_and_optimize.ipynb`**.

---

## 🛠 Công nghệ sử dụng

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)
![Pandas](https://img.shields.io/badge/Pandas-2.x-purple?logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?logo=googlecolab)

---

## 👤 Tác giả

**Nguyen Le Duc Tue**  
🔗 GitHub: [@AIVIETNAM-AIO-NguyenTue](https://github.com/AIVIETNAM-AIO-NguyenTue)  
📚 Chương trình: AIO 2026 — Artificial Intelligence Vietnam

---

## 📄 License

MIT License © 2026 AIVIETNAM-AIO-NguyenTue
