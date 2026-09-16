import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_FILE = "sinhala_call_center_final_1000.csv"
OUTPUT_DIR = "trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Text Parsing & Speaker Separation
# -----------------------------------------------------------------------------
def extract_speaker_text(full_transcript: str):
    """
    Extracts customer utterances and agent utterances separately
    from the single-line pipe-delimited transcript.
    """
    if not isinstance(full_transcript, str):
        return "", ""
    
    parts = full_transcript.split("|")
    customer_dialogues = []
    agent_dialogues = []

    for segment in parts:
        clean_seg = segment.strip()
        if clean_seg.startswith("Customer:"):
            text = clean_seg.replace("Customer:", "").strip()
            customer_dialogues.append(text)
        elif clean_seg.startswith("Agent:"):
            text = clean_seg.replace("Agent:", "").strip()
            agent_dialogues.append(text)

    # Fallback to full transcript if parsing fails
    customer_combined = " ".join(customer_dialogues) if customer_dialogues else full_transcript
    agent_combined = " ".join(agent_dialogues) if agent_dialogues else full_transcript

    return customer_combined, agent_combined

def sanitize_text(text: str) -> str:
    """Removes unwanted formatting symbols while preserving Sinhala Unicode."""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s\u0D80-\u0DFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------------------------------------------------------
# 2. Load and Prepare Dataset
# -----------------------------------------------------------------------------
print(f"Loading dataset: {DATASET_FILE}...")
df = pd.read_csv(DATASET_FILE, encoding="utf-8-sig")
print(f"Total calls loaded: {len(df)}")

# Extract speaker-specific texts
df["customer_text"], df["agent_text"] = zip(*df["transcript"].apply(extract_speaker_text))
df["customer_text"] = df["customer_text"].apply(sanitize_text)
df["agent_text"] = df["agent_text"].apply(sanitize_text)

# -----------------------------------------------------------------------------
# 3. Model Definition & Training Configuration
# -----------------------------------------------------------------------------
# Map target labels to the relevant speaker context
tasks = [
    {
        "target": "sentiment_label",
        "feature_col": "customer_text",
        "description": "Customer Sentiment Analysis"
    },
    {
        "target": "politeness",
        "feature_col": "agent_text",
        "description": "Agent Politeness Evaluation"
    },
    {
        "target": "problem_understanding",
        "feature_col": "agent_text",
        "description": "Problem Understanding Evaluation"
    },
    {
        "target": "solution_quality",
        "feature_col": "agent_text",
        "description": "Solution Quality Evaluation"
    }
]

benchmark_results = []

for task in tasks:
    target_name = task["target"]
    feature_name = task["feature_col"]
    desc = task["description"]

    print("\n" + "=" * 60)
    print(f"TRAINING: {desc} (Target: {target_name})")
    print(f"Using Feature Source: '{feature_name}'")
    print("=" * 60)

    X = df[feature_name]
    y = df[target_name]

    # Stratified split ensures balanced class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # NLP Pipeline: TF-IDF Vectorizer + Multinomial Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),        # Capture unigrams and bigrams
            max_features=5000,         # Cap vocabulary to prevent overfitting
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',   # Handle minor class skews
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc:.4f} | Macro F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save trained pipeline artifact
    model_path = os.path.join(OUTPUT_DIR, f"{target_name}_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline artifact to: {model_path}")

    benchmark_results.append({
        "Task": desc,
        "Target Label": target_name,
        "Accuracy (%)": round(acc * 100, 2),
        "Macro F1": round(f1, 4)
    })

# -----------------------------------------------------------------------------
# 4. Summary Table
# -----------------------------------------------------------------------------
print("\n" + "#" * 60)
print("FINAL BASELINE MODEL EVALUATION SUMMARY")
print("#" * 60)
summary_df = pd.DataFrame(benchmark_results)
print(summary_df.to_string(index=False))