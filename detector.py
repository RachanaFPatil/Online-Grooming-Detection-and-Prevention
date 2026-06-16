# -*- coding: utf-8 -*-
"""
SafeGuard Detection Pipeline
=============================
STEP 1 — Train and save (run ONCE):
    python safeguard_detector.py --train --csv Combined_data.csv

STEP 2 — Verify it works:
    python safeguard_detector.py --test

STEP 3 — app.py loads it automatically on startup. Done.

Requirements:
    pip install transformers datasets accelerate scikit-learn torch emoji nudenet joblib seaborn matplotlib
"""

import os, re, argparse, torch, joblib, emoji, numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════
#  SAVED FILE PATHS
# ═══════════════════════════════════════════════════════
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, "safeguard_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "safeguard_tokenizer")
META_PATH     = os.path.join(BASE_DIR, "safeguard_meta.pkl")
MODEL_NAME    = "distilbert-base-uncased"
MAX_LEN       = 128

# ═══════════════════════════════════════════════════════
#  RUNTIME GLOBALS
# ═══════════════════════════════════════════════════════
_model          = None
_tokenizer      = None
_device         = None
_nude_detector  = None
_labels_flipped = False

# ═══════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════
def clean_message(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '[url]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ═══════════════════════════════════════════════════════
#  EMOJI SCORER
# ═══════════════════════════════════════════════════════
ROMANTIC = {'❤️','😘','😍','💋','💕','💞'}
SEXUAL   = {'🔥','🥵','😏','😈','🍑','🍆','🫦'}
SECRECY  = {'🤫','🫣'}
MEDIA    = {'📸'}
SAFE     = {'😊','😂','👍','🎮','🎉','😄','🙂','😁'}

def get_emoji_score(message: str) -> float:
    emojis_found = [e["emoji"] for e in emoji.emoji_list(message)]
    if not emojis_found:
        return 0.0
    score = 0.0
    for em in emojis_found:
        if   em in SEXUAL:   score += 0.8
        elif em in SECRECY:  score += 0.7
        elif em in MEDIA:    score += 0.6
        elif em in ROMANTIC: score += 0.5
        elif em in SAFE:     score += 0.0
    return round(min(score / len(emojis_found), 1.0), 3)

# ═══════════════════════════════════════════════════════
#  IMAGE SCORER — NudeNet loaded lazily (only when needed)
# ═══════════════════════════════════════════════════════
HIGH_RISK_CLASSES = {
    'EXPOSED_GENITALIA_F', 'EXPOSED_GENITALIA_M',
    'EXPOSED_BREAST_F', 'EXPOSED_BUTTOCKS',
    'EXPLICIT_SEXUAL_ACTIVITY'
}
MEDIUM_RISK_CLASSES = {
    'MALE_BREAST_EXPOSED', 'FEMALE_BREAST_COVERED',
    'FEMALE_GENITALIA_COVERED', 'ARMPITS_EXPOSED'
}

def get_image_score(image_path: str) -> float:
    """Score an image file. Returns 0.0 if path is None or file missing."""
    if not image_path or not os.path.exists(str(image_path)):
        return 0.0
    global _nude_detector
    try:
        if _nude_detector is None:
            from nudenet import NudeDetector
            _nude_detector = NudeDetector()
        detections = _nude_detector.detect(image_path)
        if not detections:
            return 0.0
        high = [d['score'] for d in detections if d['class'] in HIGH_RISK_CLASSES]
        med  = [d['score'] * 0.5 for d in detections if d['class'] in MEDIUM_RISK_CLASSES]
        if high: return round(max(high), 4)
        if med:  return round(max(med),  4)
        return 0.0
    except Exception as e:
        print(f"[IMAGE SCORER] Error: {e}")
        return 0.0

# ═══════════════════════════════════════════════════════
#  LOAD SAVED MODEL — called once at app.py startup
# ═══════════════════════════════════════════════════════
def load_model():
    """Load the saved fine-tuned DistilBERT. Call once before analyze_message()."""
    global _model, _tokenizer, _device, _labels_flipped

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"\n[SafeGuard] Trained model not found at: {MODEL_DIR}\n"
            f"Run training first:\n"
            f"    python safeguard_detector.py --train --csv Combined_data.csv\n"
        )

    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_DIR)
    _model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()

    if os.path.exists(META_PATH):
        meta = joblib.load(META_PATH)
        _labels_flipped = meta.get("labels_flipped", False)

    print(f"[SafeGuard] ✓ Model loaded on {_device} | labels_flipped={_labels_flipped}")

# ═══════════════════════════════════════════════════════
#  TEXT SCORER
# ═══════════════════════════════════════════════════════
def get_text_score(message: str) -> float:
    """Returns grooming probability 0.0–1.0 from fine-tuned DistilBERT."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    inputs = _tokenizer(
        clean_message(message),
        return_tensors='pt', truncation=True, max_length=MAX_LEN
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    grooming_prob = probs[1]  # index 1 = GROOMING
    if _labels_flipped:
        grooming_prob = 1.0 - grooming_prob
    return round(grooming_prob, 4)

# ═══════════════════════════════════════════════════════
#  MAIN ANALYZE FUNCTION — called by app.py per message
# ═══════════════════════════════════════════════════════
def analyze_message(message: str, image_path: str = None) -> dict:
    """
    Full SafeGuard pipeline — text + emoji + image.

    Args:
        message    : raw chat message string (text + any emojis)
        image_path : local path to image file (optional)

    Returns:
        {
          'score'      : float 0.0–1.0,
          'risk_level' : 'SAFE' | 'SUSPICIOUS' | 'HIGH_RISK' | 'CRITICAL',
          'action'     : str description,
          'breakdown'  : { text_score, emoji_score, image_score }
        }
    """
    text_score  = get_text_score(message)
    emoji_score = get_emoji_score(message)
    image_score = get_image_score(image_path) if image_path else 0.0

    # Weighted fusion
    if image_path:
        final = text_score * 0.60 + emoji_score * 0.15 + image_score * 0.25
    else:
        final = text_score * 0.80 + emoji_score * 0.20

    final = round(min(max(final, 0.0), 1.0), 4)

    if   final < 0.40: level = 'SAFE';       action = 'Silent monitoring. Detection app invisible.'
    elif final < 0.60: level = 'SUSPICIOUS';  action = 'Flag internally. Increase monitoring.'
    elif final < 0.80: level = 'HIGH_RISK';   action = 'Pop-up shown to child. Chatbot offered.'
    else:              level = 'CRITICAL';    action = 'Sandbox + pop-up + silent parent email.'

    return {
        'score'     : final,
        'risk_level': level,
        'action'    : action,
        'breakdown' : {
            'text_score' : round(text_score,  4),
            'emoji_score': round(emoji_score, 4),
            'image_score': round(image_score, 4),
        }
    }

# ═══════════════════════════════════════════════════════
#  TRAINING — run ONCE, saves everything to disk
# ═══════════════════════════════════════════════════════
def train_and_save(csv_path: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        accuracy_score, classification_report, confusion_matrix
    )
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        TrainingArguments, Trainer
    )
    from datasets import Dataset

    print("=" * 55)
    print("  SafeGuard — Training DistilBERT (runs ONCE)")
    print(f"  Dataset : {csv_path}")
    print("=" * 55)

    # ── Load & clean ──
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    df = df[df['labels'].astype(str).isin(['0','1'])].copy()
    df['labels']     = df['labels'].astype(int)
    df['text']       = df['text'].astype(str).str.strip()
    df               = df[df['text'].str.len() > 3].reset_index(drop=True)
    df['text_clean'] = df['text'].apply(clean_message)

    print(f"\nTotal  : {len(df)} | Grooming: {(df['labels']==1).sum()} | Normal: {(df['labels']==0).sum()}")

    # Sample messages preview
    print("\n── Sample grooming messages ──")
    for t in df[df['labels']==1]['text'].sample(min(3, (df['labels']==1).sum()), random_state=42):
        print(f"  > {t[:90]}")
    print("\n── Sample normal messages ──")
    for t in df[df['labels']==0]['text'].sample(min(3, (df['labels']==0).sum()), random_state=42):
        print(f"  > {t[:90]}")

    # ── Split 70/15/15 ──
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['labels'], random_state=42)
    val_df,  test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df['labels'], random_state=42)
    print(f"\nTrain:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")

    # ── Tokenize ──
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if not torch.cuda.is_available():
        print("⚠  No GPU — training will be slow on CPU. Consider using Google Colab T4.")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch['text_clean'], truncation=True, padding='max_length', max_length=MAX_LEN)

    def make_ds(split_df):
        ds = Dataset.from_dict({
            'text_clean': split_df['text_clean'].tolist(),
            'label':      split_df['labels'].tolist()
        })
        ds = ds.map(tokenize_fn, batched=True)
        ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
        return ds

    train_ds = make_ds(train_df)
    val_ds   = make_ds(val_df)
    test_ds  = make_ds(test_df)
    print("Tokenization complete ✓")

    # ── Model ──
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0:'NORMAL', 1:'GROOMING'},
        label2id={'NORMAL':0, 'GROOMING':1}
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy' : round(accuracy_score(labels, preds), 4),
            'f1'       : round(f1_score(labels, preds, pos_label=1), 4),
            'precision': round(precision_score(labels, preds, pos_label=1, zero_division=0), 4),
            'recall'   : round(recall_score(labels, preds, pos_label=1, zero_division=0), 4),
        }

    use_fp16 = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir                  = os.path.join(BASE_DIR, 'safeguard-checkpoints'),
        num_train_epochs            = 1,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size  = 64,
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        learning_rate               = 2e-5,
        lr_scheduler_type           = 'cosine',
        eval_strategy               = 'epoch',
        save_strategy               = 'epoch',
        load_best_model_at_end      = True,
        metric_for_best_model       = 'f1',
        fp16                        = use_fp16,
        logging_steps               = 50,
        report_to                   = 'none',
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nTraining started — watch F1 improve each epoch...\n")
    trainer.train()

    # ── Sanity check — detect label flip ──
    model.eval()
    _known = 'have you ever done anal? send me a pic, keep it secret'
    _enc   = tokenizer(_known, return_tensors='pt', truncation=True, max_length=MAX_LEN)
    _enc   = {k: v.to(model.device) for k, v in _enc.items()}
    with torch.no_grad():
        _logits = model(**_enc).logits
    _probs = torch.softmax(_logits, dim=-1)[0].cpu().tolist()
    _pred  = model.config.id2label[int(torch.argmax(_logits))]
    labels_flipped = (_pred == 'NORMAL' and _probs[0] > 0.7)
    print(f"\nSanity check → pred: {_pred} | NORMAL: {_probs[0]:.4f} | GROOMING: {_probs[1]:.4f}")
    print(f"Labels flipped: {labels_flipped}")

    # ── Save model + tokenizer + meta ──
    print("\nSaving model to disk...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    joblib.dump({"labels_flipped": labels_flipped}, META_PATH)

    print(f"\n✓ Model saved     → {MODEL_DIR}/")
    print(f"✓ Tokenizer saved → {TOKENIZER_DIR}/")
    print(f"✓ Meta saved      → {META_PATH}")

    # ── Test set evaluation ──
    print("\n=== Test Set Results ===")
    preds_out = trainer.predict(test_ds)
    preds  = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    print(classification_report(labels, preds, target_names=['Normal','Grooming']))

    # Confusion matrix (saved as PNG)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal','Grooming'],
                    yticklabels=['Normal','Grooming'], ax=ax)
        ax.set_title('SafeGuard — DistilBERT Confusion Matrix')
        ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        plt.tight_layout()
        out_png = os.path.join(BASE_DIR, 'confusion_matrix.png')
        plt.savefig(out_png, dpi=150)
        print(f"\nConfusion matrix saved → {out_png}")
    except Exception as e:
        print(f"(Skipped confusion matrix plot: {e})")

    print("\n" + "="*55)
    print("  Training complete!")
    print("  You never need to retrain unless the dataset changes.")
    print("  app.py will now load the saved model on every startup.")
    print("="*55)

# ═══════════════════════════════════════════════════════
#  QUICK TEST — verify saved model works
# ═══════════════════════════════════════════════════════
def run_test():
    load_model()
    tests = [
        "Hey how was school today?",
        "don't tell your parents about us 🤫 it's our little secret ❤️",
        "send me a pic, you look so mature 😘",
        "are you home alone right now?",
        "haha that game was fun 😂",
        "have you ever done anything like that?",
        "when are we meeting? 😉",
        "can I get your number",
    ]
    print("\n=== SafeGuard — Inference Test ===\n")
    for msg in tests:
        r   = analyze_message(msg)
        bar = '█' * int(r['score'] * 25)
        print(f"[{r['risk_level']:10s}] {r['score']:.3f}  {bar}")
        print(f"  msg   : {msg}")
        print(f"  layers: text={r['breakdown']['text_score']}  "
              f"emoji={r['breakdown']['emoji_score']}")
        print()

# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeGuard Detection Pipeline")
    parser.add_argument("--train", action="store_true", help="Train and save model (run once)")
    parser.add_argument("--csv",   type=str, default="Combined_data.csv", help="Path to dataset CSV")
    parser.add_argument("--test",  action="store_true", help="Load saved model and run quick test")
    args = parser.parse_args()

    if args.train:
        train_and_save(args.csv)
    elif args.test:
        run_test()
    else:
        parser.print_help()