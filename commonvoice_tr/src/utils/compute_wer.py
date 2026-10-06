import re
import string
import jiwer

def normalize_turkish_text(text: str) -> str:
    """
    Normalizes Turkish text for ASR WER evaluation.
    - Turkish specific lowercasing (I -> ı, İ -> i)
    - Removes punctuation and extra whitespace
    - Preserves Turkish characters (ç, ğ, ı, ö, ş, ü)
    """
    if not text:
        return ""
    
    # Custom Turkish lowercasing
    text = text.replace("I", "ı").replace("İ", "i")
    text = text.lower()
    
    # Remove punctuation
    punct = string.punctuation + "“”‘’«»…"
    text = re.sub(f"[{re.escape(punct)}]", " ", text)
    
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def compute_wer_cer(predictions: list, references: list):
    """
    Computes WER and CER after Turkish normalization.
    """
    norm_preds = [normalize_turkish_text(p) for p in predictions]
    norm_refs = [normalize_turkish_text(r) for r in references]
    
    # Filter out empty references if any
    valid_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if len(r.strip()) > 0]
    if not valid_pairs:
        return {"wer": 0.0, "cer": 0.0}
    
    norm_preds, norm_refs = zip(*valid_pairs)
    
    wer = jiwer.wer(list(norm_refs), list(norm_preds))
    cer = jiwer.cer(list(norm_refs), list(norm_preds))
    
    return {
        "wer": float(wer) * 100.0,
        "cer": float(cer) * 100.0,
        "normalized_preds": norm_preds,
        "normalized_refs": norm_refs,
    }
