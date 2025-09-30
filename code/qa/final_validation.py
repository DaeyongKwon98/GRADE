import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import re
import os

load_dotenv()
client = OpenAI()
folder_path = os.getenv("FOLDER_PATH", "demo")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

validation_prompt = """You are an AI assistant tasked with reviewing question and answer pairs for ambiguity or vagueness.
Your goal is to evaluate whether each pair is clear and self-contained — that is, whether it can be understood without relying on external or missing context.

Use the following criteria to make your judgment:
The question and answer must be decontextualized — meaning they should be understandable on their own, without requiring additional background information.
If the answer includes vague references such as "other countries," "certain individuals," or "this technology," and the question does not provide enough information to specify what these refer to, then it is considered ambiguous.
Similarly, if the question uses pronouns or context-dependent expressions like "he," "they," "this," or "that" without clearly indicating the referent, the pair is not decontextualized and should be marked as ambiguous.

Based on these criteria:
If the question-answer pair is decontextualized and unambiguous, output True.
If it relies on missing context or includes vague or ambiguous expressions, output False.

Output format:
True / False"""


for hop in range(2,6):
    qa_path = f"{folder_path}/qa_{hop}hop_400sample.csv"
    if not os.path.exists(qa_path) or os.path.getsize(qa_path) == 0:
        continue
    print(f"Processing {hop}hop...")
    try:
        qa = pd.read_csv(qa_path)
    except Exception:
        continue
    
    result = []
    for i, row in tqdm(qa.iterrows(), total=len(qa)):
        user_prompt = f"Question: {row['question']}\nAnswer: {row['answer']}"

        completion = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "assistant", "content": validation_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2000,
        )

        response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            completion = client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "assistant", "content": validation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=3000,
            )
            response = completion.choices[0].message.content.strip()
        
        if 'true' in response.lower():
            result.append({'is_decontextualized': True})
        else:
            result.append({'is_decontextualized': False})
    
    pd.DataFrame(result).to_csv(f"{folder_path}/qa_{hop}hop_400sample_decontextualized.csv", index=False)

    meta_path = f"{folder_path}/{hop}hop_qa_400sample.csv"
    dec_path = f"{folder_path}/qa_{hop}hop_400sample_decontextualized.csv"
    if not (os.path.exists(meta_path) and os.path.exists(dec_path)):
        continue
    try:
        metadata = pd.read_csv(meta_path)
        is_valid = pd.read_csv(dec_path)
        qa = pd.read_csv(qa_path)
    except Exception:
        continue

    # True인 행들의 인덱스 구하기
    indices = is_valid.index[is_valid['is_decontextualized']].tolist()
    print(len(indices), indices)

    # 해당 인덱스에 해당하는 df의 행만 선택
    selected_rows = pd.concat([metadata, qa], axis=1).loc[indices]
    selected_rows.to_csv(f"{folder_path}/qa_{hop}hop_400sample_final.csv", index=False)
    print(f"Final {len(selected_rows)} rows saved to qa_final.csv")
    