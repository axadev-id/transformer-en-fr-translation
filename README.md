# TUGAS EKSPLORASI Deep Learning
## Nama Kelompok :
- Fajrul Ramadhana Aqsa-122140118
- Ichsan Kuntadi Baskara-122140117
- Mychael Daniel N-122140104

# Transformer English-French Translation

Implementasi Neural Machine Translation (NMT) menggunakan arsitektur Transformer untuk menerjemahkan teks dari Bahasa Inggris ke Bahasa Prancis. Project ini dibuat sebagai bagian dari pembelajaran Deep Learning dengan fokus pada mekanisme attention dan sequence-to-sequence models.

## 📋 Deskripsi Project
Project ini mengimplementasikan model Transformer dari scratch menggunakan PyTorch untuk tugas machine translation. Model dilatih pada dataset parallel English-French dengan ~137,860 pasangan kalimat. Implementasi mencakup:

- **Positional Encoding** untuk memberikan informasi posisi token
- **Multi-Head Attention** mechanism (8 attention heads)
- **Encoder-Decoder Architecture** (3 layers each)
- **Masking mechanisms** (padding mask & causal mask) untuk training yang proper
- **Autoregressive Evaluation** untuk mengukur performa inference yang realistis

## 🎯 Fitur Utama

- ✅ Dataset loading yang robust (menghindari CSV parsing errors)
- ✅ Word-level tokenization dengan vocabulary building
- ✅ Proper masking implementation:
  - Padding masks untuk encoder & decoder
  - Causal mask untuk mencegah attention ke token masa depan
- ✅ Teacher forcing training dengan validation metrics
- ✅ Greedy decoding untuk inference
- ✅ Evaluasi ganda: teacher-forcing accuracy & autoregressive metrics
- ✅ Dokumentasi lengkap dengan analisis exposure bias

## 📊 Hasil Training

**Model Configuration:**
- d_model: 256
- Attention heads: 8
- Encoder/Decoder layers: 3
- Vocabulary size: ~20k-30k tokens (source & target)
- Batch size: 64
- Optimizer: Adam (lr=1e-4)

**Hasil setelah 1 Epoch:**

| Metric | Value | Keterangan |
|--------|-------|------------|
| Validation Loss | 0.7250 | Model mulai belajar pola translasi |
| Validation Accuracy | 79.95% | Token-level accuracy dengan teacher forcing |
| Greedy Exact Match | ~0-2% | Persentase kalimat yang 100% benar |
| Token Overlap | ~15-30% | Persentase token yang cocok posisi |

**Catatan Penting:** Gap besar antara validation accuracy (79.95%) dan greedy performance (<5%) menunjukkan **exposure bias** - fenomena umum dalam seq2seq models di mana model tidak terbiasa dengan error sendiri saat inference.

## 🚀 Cara Menggunakan

### Prerequisites

```bash
pip install torch pandas numpy tqdm
```

### Running the Notebook

1. Clone repository:
```bash
git clone https://github.com/axadev-id/transformer-en-fr-translation.git
cd transformer-en-fr-translation
```

2. Pastikan file dataset tersedia:
   - `small_vocab_en.csv` - Dataset Bahasa Inggris
   - `small_vocab_fr.csv` - Dataset Bahasa Prancis

3. Buka notebook:
```bash
jupyter notebook Transformer_Translation_fixed.ipynb
```

4. Jalankan cell secara berurutan dari atas ke bawah

### Quick Test

Setelah training, gunakan fungsi `translate_sentence()` untuk test:

```python
translate_sentence(model, "Hello, how are you?", src_stoi, tgt_stoi, tgt_itos)
```

## 📁 Struktur File

```
transformer-en-fr-translation/
├── Transformer_Translation_fixed.ipynb    # Notebook utama (recommended)
├── Transformer_Translation.ipynb          # Versi original
├── Transformer_Translation (1).ipynb      # Versi eksperimen
├── transformer_translation.py             # Script Python (jika ada)
├── small_vocab_en.csv                     # Dataset English (~137k sentences)
├── small_vocab_fr.csv                     # Dataset French (~137k sentences)
└── README.md                              # Dokumentasi ini
```

## 🔬 Analisis: Exposure Bias Problem

### Kenapa Validation Accuracy Tinggi tapi Translation Jelek?

**Teacher Forcing (Training):**
- Model diberi ground truth tokens sebagai input
- Accuracy tinggi (79.95%) karena context selalu sempurna
- Model tidak belajar recovery dari error

**Autoregressive (Inference):**
- Model generate token sendiri tanpa ground truth
- Error di awal terakumulasi (error propagation)
- Hasil: repetisi, degradasi, atau nonsense output

**Contoh:**
```
Input:    "Tom was here yesterday"
Expected: "Tom était ici hier"
Actual:   "je suis de la de la maison" (repetisi + salah konteks)
```

### Solusi Exposure Bias

1. **Scheduled Sampling** - Secara bertahap gunakan prediksi model saat training
2. **Beam Search** - Eksplorasi multiple kandidat saat inference
3. **Label Smoothing** - Regularization untuk mengurangi overconfidence
4. **Training lebih lama** - 10-50 epoch untuk konvergensi lebih baik

## 📈 Roadmap Perbaikan

### Short-term (Quick Wins)
- [ ] Train hingga 10-20 epoch
- [ ] Implementasi learning rate scheduler
- [ ] Gradient clipping untuk stabilitas training

### Medium-term (Better Performance)
- [ ] Implementasi Beam Search decoding
- [ ] Scheduled Sampling untuk kurangi exposure bias
- [ ] Label Smoothing regularization
- [ ] BLEU score evaluation

### Long-term (Production Quality)
- [ ] Subword tokenization (BPE/SentencePiece)
- [ ] Dataset lebih besar (1M+ pairs)
- [ ] Model capacity lebih besar (d_model=512, nlayers=6)
- [ ] Pre-training atau transfer learning dari multilingual models

## 📖 Penjelasan Teknis

### Arsitektur Model

**PositionalEncoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**TransformerMT:**
```
Input → Embedding → Positional Encoding → Transformer → Linear → Output
```

**Masking:**
1. **Padding Mask**: Boolean mask untuk ignore padding tokens
   - Shape: (batch_size, seq_len)
   - True = padding, False = valid token

2. **Causal Mask**: Upper triangular mask untuk autoregressive decoding
   - Mencegah attention ke posisi masa depan
   - Shape: (seq_len, seq_len)

### Training Process

```python
for epoch in epochs:
    for batch in train_loader:
        # Generate masks
        tgt_mask = generate_square_subsequent_mask(tgt_len)
        src_padding_mask = (src == pad_idx)
        tgt_padding_mask = (tgt == pad_idx)
        
        # Forward pass dengan masks
        output = model(src, tgt, 
                      src_key_padding_mask=src_padding_mask,
                      tgt_key_padding_mask=tgt_padding_mask,
                      tgt_mask=tgt_mask)
        
        # Loss & backward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## 🎓 Learning Outcomes

Project ini memberikan pemahaman mendalam tentang:

1. **Transformer Architecture** - Encoder-decoder, attention mechanism, positional encoding
2. **Sequence-to-Sequence Models** - Translation task, autoregressive generation
3. **Training Techniques** - Teacher forcing, masking, loss functions
4. **Evaluation Metrics** - Teacher-forcing accuracy vs autoregressive metrics
5. **Common Pitfalls** - Exposure bias, metric misinterpretation, overfitting

## 📚 Referensi

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Scheduled Sampling for Sequence Prediction](https://arxiv.org/abs/1506.03099)

## 🤝 Kontribusi

Project ini dibuat untuk tujuan pembelajaran. Saran dan perbaikan sangat diterima! Silakan:

1. Fork repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add some improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request


## 👤 Author

**axadev-id**
- GitHub: [@axadev-id](https://github.com/axadev-id)
- GitHub: [@mychaeldaniel122140104](https://github.com/mychaeldaniel122140104)
- GitHub: [@ichsank18](https://github.com/ichsank18)
- Repository: [transformer-en-fr-translation](https://github.com/axadev-id/transformer-en-fr-translation)

## 🙏 Acknowledgments

- Dataset: English-French parallel corpus
- Framework: PyTorch
- Inspiration: "Attention is All You Need" paper
- Community: PyTorch & Deep Learning Indonesia

---

**Status Project:** 🟡 In Development (1 Epoch trained, baseline model ready)


