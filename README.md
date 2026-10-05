# 🎙️ SLAM-ASR: Speech Recognition via Speech-Language Model

Bu proje, **SLAM-LLM** (*[SLAM-LLM: Speech-Language Critical Evaluation](https://arxiv.org/abs/2402.08846)*) çalışmasının **LibriSpeech Otomatik Konuşma Tanıma (ASR)** modülünün bağımsız, temiz ve Google Colab uyumlu bir uygulamasıdır.

---

## 📊 Makale Performans İddiaları (LibriSpeech Benchmarks)

Makalede sunulan SLAM-ASR mimarisi, dondurulmuş (frozen) bir Konuşma Kodlayıcısı (Speech Encoder), eğitilebilir küçük bir Lineer Projektör (~18M - 21M parametre) ve dondurulmuş bir Büyük Dil Modelinden (Vicuna-7B-v1.5) oluşur.

| Speech Encoder | Projector | LLM | test-clean WER (%) | test-other WER (%) |
|---|---|---|---|---|
| **WavLM-large** | Linear (~18.88M) | Vicuna-7B-v1.5 | **2.28%** | **4.78%** |
| **HuBERT-xtralarge** | Linear (~21.50M) | Vicuna-7B-v1.5 | **1.84%** | **3.39%** |

---

## ⚡ Quick Start: Google Colab Uyumlu Notebook

Projede hazırlanan **`slam_asr_librispeech_colab.ipynb`** dosyası ile modeli hiçbir yerel kurulum yapmadan doğrudan **Google Colab** üzerinde (T4 GPU veya A100 GPU) test edebilir ve performans rakamlarını doğrulayabilirsiniz.

1. `slam_asr_librispeech_colab.ipynb` dosyasını Google Colab'e yükleyin veya GitHub üzerinden açın.
2. Colab GPU çalışma zamanını seçin (**Runtime -> Change runtime type -> GPU (T4 / L4 / A100)**).
3. Notebook hücresini sırayla çalıştırarak test-clean veri seti üzerinde çıkarım yapın ve WER sonucunu görüntüleyin.

---

## 💻 Yerel (Local) Kurulum ve Çalıştırma

### 1. Bağımlılıkların Yüklenmesi

```bash
pip install -r requirements.txt
```

### 2. Ön Eğitilmiş Model Ağırlıklarının İndirilmesi

- **WavLM-Large Encoder**: [WavLM-Large.pt](https://github.com/microsoft/unilm/raw/master/wavlm/WavLM-Large.pt)
- **WavLM Linear Projector Checkpoint**: [Google Drive Link](https://drive.google.com/file/d/1cLNuMR05oXxKj8M_Z3yAZ5JHJ06ybIHp/view?usp=sharing) (`file_id: 1cLNuMR05oXxKj8M_Z3yAZ5JHJ06ybIHp`)
- **HuBERT Linear Projector Checkpoint**: [Google Drive Link](https://drive.google.com/file/d/1Np7EjMYSZCl7M6Q92pt_MvOSSX6ggJPA/view?usp=drive_link) (`file_id: 1Np7EjMYSZCl7M6Q92pt_MvOSSX6ggJPA`)

```bash
mkdir -p checkpoints

# WavLM-Large Encoder Ağırlığı
wget -O checkpoints/WavLM-Large.pt https://github.com/microsoft/unilm/raw/master/wavlm/WavLM-Large.pt

# WavLM Linear Projector Checkpoint
gdown https://drive.google.com/uc?id=1cLNuMR05oXxKj8M_Z3yAZ5JHJ06ybIHp -O checkpoints/wavlm_linear_projector.pt
```

### 3. LibriSpeech Veri Hazırlığı

Veri setinizi aşağıdaki formatta bir `.jsonl` dosyasına dönüştürün:

```json
{"key": "1001-134707-0000_ASR", "source": "/path/to/librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer..."}
```

### 4. Çıkarım (Inference / Decoding)

WavLM-Large + Vicuna-7B modeli ile çıkarım yapmak için:

```bash
python examples/asr_librispeech/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++model_config.llm_name="vicuna-7b-v1.5" \
    ++model_config.llm_path="lmsys/vicuna-7b-v1.5" \
    ++model_config.llm_dim=4096 \
    ++model_config.encoder_name=wavlm \
    ++model_config.normalize=true \
    ++dataset_config.normalize=true \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path="checkpoints/WavLM-Large.pt" \
    ++model_config.encoder_dim=1024 \
    ++model_config.encoder_projector=linear \
    ++dataset_config.dataset=speech_dataset \
    ++dataset_config.val_data_path="data/librispeech_test_clean.jsonl" \
    ++dataset_config.input_type=raw \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batching_strategy=custom \
    ++train_config.val_batch_size=1 \
    ++train_config.output_dir="output/decode_results" \
    ++decode_log="output/decode_results/decode_test_clean_beam4" \
    ++ckpt_path="checkpoints/wavlm_linear_projector.pt"
```

### 5. WER (Word Error Rate) Hesaplama

```bash
python examples/asr_librispeech/eval_wer.py \
    output/decode_results/decode_test_clean_beam4_pred \
    output/decode_results/decode_test_clean_beam4_gt \
    output/decode_results/wer_detail.txt
```

---

## 📁 Proje Dizin Yapısı

```
.
├── README.md                           # Proje dokümantasyonu ve kullanım kılavuzu
├── requirements.txt                    # Gerekli Python kütüphaneleri
├── slam_asr_librispeech_colab.ipynb    # Google Colab interaktif notebook'u
├── src/                                # SLAM-LLM temel modülleri (models, datasets, utils)
└── examples/
    └── asr_librispeech/                # ASR yapılandırmaları ve çıkarım betikleri
        ├── conf/                       # Hydra yapılandırma dosyaları (prompt.yaml)
        ├── scripts/                    # Shell çıkarım betikleri
        ├── asr_config.py               # Veri ve model dataclass tanımları
        ├── inference_asr_batch.py      # Toplu çıkarım betiği
        └── eval_wer.py                 # WER hesaplama yardımcısı
```

---

## 🔗 Referanslar

- **Makale**: [SLAM-LLM: Speech-Language Critical Evaluation (arXiv:2402.08846)](https://arxiv.org/abs/2402.08846)
- **Orijinal Repository**: [X-LANCE / SLAM-LLM (GitHub)](https://github.com/X-LANCE/SLAM-LLM)
