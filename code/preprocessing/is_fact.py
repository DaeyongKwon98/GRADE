import os
import pandas as pd
from tqdm import tqdm

FAST_DEMO = os.getenv("FAST_DEMO")

# Lazy import transformers only if needed
pipe = None
if not FAST_DEMO:
    from transformers import pipeline  # type: ignore
    pipe = pipeline("text-classification", model="lighteternal/fact-or-opinion-xlmr-el")

folder_path = os.getenv("FOLDER_PATH", "demo")

# Read input CSV
df = pd.read_csv(f"{folder_path}/sentence.csv")

def classify_fact_or_opinion(text):
    if FAST_DEMO:
        return True
    # Limit input length (XLMR supports up to 512 tokens)
    result = pipe(text[:512])[0]
    print(result)
    # Interpret 'LABEL_1' as 'fact'
    return result['label'] == 'LABEL_1'

# Progress bar
tqdm.pandas(desc="Processing sentences")

# Add 'is_fact' column
df['is_fact'] = df['text'].progress_apply(classify_fact_or_opinion)

# Save results
df.to_csv(f"{folder_path}/sentences_isfact.csv", index=False)