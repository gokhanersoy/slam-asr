# 🎙️ SLAM-ASR Türkçe: Common Voice Benchmark

Bu proje, **SLAM-LLM (Speech-Language Model Framework)** çalışmasını ([arXiv:2402.08846](https://arxiv.org/pdf/2402.08846)) birebir takip ederek **Common Voice Türkçe** veri setinde State-of-the-Art (SOTA) ASR başarımına ulaşmak için geliştirilmiştir.

---

## ⚡ Özellikler ve Çalışma Mantığı

- **Google Drive Gerektirmez:** Kodlar doğrudan GitHub üzerinden Colab oturumuna klonlanır.
- **Otomatik Hugging Face Entegrasyonu:** Eğitim bittiğinde eğitilen model ağırlıkları (Projector + LoRA) otomatik olarak kendi Hugging Face Hub hesabınıza aktarılır (`push_to_hub.py`).
- **Maksimum Performans:** Veri seti işleme ve önbellekleme Colab'ın yüksek hızlı yerel SSD diski üzerinde gerçekleşir.

---

## 📐 Mimari ve Metodoloji

1. **Konuşma Kodlayıcı (Speech Encoder):**
   - `openai/whisper-large-v3` (128 mel kanalı).
   - Eğitim süresince **dondurulmuştur (Frozen)**.

2. **Hizalama Katmanı (Linear Projector):**
   - 5x Frame Concatenation + 2-Katmanlı MLP (`Linear(1280 * 5, 2048) -> ReLU -> Linear(2048, LLM_DIM)`).
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
├── push_to_hub.py                     # Hugging Face Hub model yükleme betiği
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

## 💻 Google Colab & Yerel Kullanım

### 1. Colab'da Çalıştırma (Tavsiye Edilen)
Notebook dosyasını açıp adımları takip edin: [`commonvoice_tr_slam_llm.ipynb`](file:///Users/gokhanersoy/Documents/GitHub/slam-asr/commonvoice_tr/commonvoice_tr_slam_llm.ipynb)

```bash
# Kodları Colab lokal diskine indirin
!git clone https://github.com/gokhanersoy/slam-asr.git
%cd /content/slam-asr/commonvoice_tr
!pip install -q -r requirements.txt

# Veriyi hazırlayın ve eğitimi başlatın
!python prepare_data.py --output_dir data
!python train.py

# Test edin ve Hugging Face Hub'a yükleyin
!python evaluate.py
!python push_to_hub.py --repo_id kullanici_adiniz/slam-asr-tr-commonvoice
```

---

## 📊 SOTA Hedefleri & Karşılaştırma (Common Voice TR)

| Model | Veri Seti | WER (%) |
| :--- | :--- | :--- |
| Whisper Large V3 (Zero-shot Baseline) | Common Voice TR | ~11.5% - 13.0% |
| Wav2Vec2-Large-TR (Fine-tuned) | Common Voice TR | ~9.2% |
| **SLAM-ASR (Whisper-v3 + Qwen2.5-7B + Linear Projector)** | Common Voice TR | **< 6.5% (Hedef)** |
