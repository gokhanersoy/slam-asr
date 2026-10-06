# 🎙️ SLAM-ASR Türkçe: Common Voice Benchmark

Bu proje, **SLAM-LLM (Speech-Language Model Framework)** çalışmasını ([arXiv:2402.08846](https://arxiv.org/pdf/2402.08846)) birebir takip ederek **Common Voice Türkçe** veri setinde State-of-the-Art (SOTA) ASR başarımına ulaşmak için geliştirilmiştir.

---

## 📐 Mimari ve Metodoloji

SLAM-LLM yaklaşımına sadık kalınarak mimari 3 ana bileşenden oluşturulmuştur:

1. **Konuşma Kodlayıcı (Speech Encoder):**
   - `openai/whisper-large-v3`
   - Mel-spektrogram (128 mel kanalı) girdisini işler.
   - Eğitim süresince **dondurulmuştur (Frozen)**.

2. **Hizalama Katmanı (Linear Projector):**
   - 5x Frame Concatenation + 2-Layer MLP Projection (`Linear(1280 * 5, 2048) -> ReLU -> Linear(2048, LLM_DIM)`).
   - Konuşma vektörlerinin zaman boyutunu 5 kat küçülterek LLM gömme (embedding) uzayına hizalar.
   - **Eğitilebilir (Trainable)**.

3. **Büyük Dil Modeli (LLM Backbone):**
   - `Qwen/Qwen2.5-7B-Instruct` (Varsayılan) veya `meta-llama/Meta-Llama-3-8B-Instruct`.
   - Türkçe dil bilgisi, morfoloji ve bağlamsal düzeltme kapasitesini doğrudan ASR deşifresine aktarır.
   - **LoRA (Low-Rank Adaptation)** ile eğitilir (r=16, alpha=32).

---

## 🚀 Dizin Yapısı

```
commonvoice_tr/
├── README.md                          # Proje açıklaması ve Türkçe ASR rehberi
├── requirements.txt                   # Gerekli bağımlılıklar
├── commonvoice_tr_slam_llm.ipynb      # Google Colab uçtan uca eğitim notebook'u
├── conf/
│   └── prompt.yaml                    # Hydra model, veri seti ve eğitim konfigürasyonu
├── src/
│   ├── models/
│   │   ├── slam_model.py              # SLAMASRModel ana sınıfı ve model_factory
│   │   ├── encoder.py                 # WhisperWrappedEncoder
│   │   └── projector.py               # EncoderProjectorConcat (5x Downsampling)
│   ├── datasets/
│   │   └── speech_dataset.py          # JSONL formatlı ses-metin veri yükleyicisi
│   └── utils/
│       └── compute_wer.py             # Türkçe karakter normalizasyonu ve WER/CER hesabı
├── prepare_data.py                    # HuggingFace Common Voice TR indirici & JSONL dönüştürücü
├── train.py                           # Fine-tuning başlatıcı betik
└── evaluate.py                        # Test seti WER/CER değerlendirme betiği
```

---

## ⚡ Hızlı Başlangıç (Google Colab)

1. Notebook dosyasını Google Colab'a yükleyin: [`commonvoice_tr_slam_llm.ipynb`](file:///Users/gokhanersoy/Documents/GitHub/slam-asr/commonvoice_tr/commonvoice_tr_slam_llm.ipynb)
2. GPU çalışma zamanını seçin (T4, V100, L4 veya A100).
3. Notebook adımlarını sırasıyla çalıştırın.

---

## 💻 Yerel (Local) Kullanım

### 1. Bağımlılıkların Kurulumu
```bash
pip install -r requirements.txt
```

### 2. Veri Setinin Hazırlanması
Mozilla Common Voice Türkçe veri setini indirip JSONL formatına dönüştürmek için:
```bash
python prepare_data.py --output_dir data
```

### 3. Model Eğitimi (Fine-Tuning)
```bash
python train.py
```
Özel parametrelerle çalıştırmak için:
```bash
python train.py train_config.batch_size_training=4 train_config.gradient_accumulation_steps=4 train_config.num_epochs=5
```

### 4. Değerlendirme (WER / CER Hesabı)
```bash
python evaluate.py --checkpoint_path checkpoints/slam_qwen_tr/best_checkpoint.pt
```

---

## 📊 SOTA Hedefleri & Karşılaştırma (Common Voice TR)

| Model | Veri Seti | WER (%) |
| :--- | :--- | :--- |
| Whisper Large V3 (Zero-shot Baseline) | Common Voice TR | ~11.5% - 13.0% |
| Wav2Vec2-Large-TR (Fine-tuned) | Common Voice TR | ~9.2% |
| **SLAM-ASR (Whisper-v3 + Qwen2.5-7B + Linear Projector)** | Common Voice TR | **< 6.5% (Hedeflenen SOTA)** |

---

## 📝 Notlar
- Qwen2.5-7B-Instruct dil modeli Türkçe karakter setini ve morfolojisini LLaMA-1/2 modellerine kıyasla çok daha verimli temsil etmektedir.
- Google Colab T4/V100 ekran kartlarında bellek optimizasyonu için 4-bit quantization (`bitsandbytes`) konfigürasyonu varsayılan olarak etkinleştirilmiştir.
