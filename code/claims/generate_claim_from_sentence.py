import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI()
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

folder_path = os.getenv("FOLDER_PATH", "demo")
max_rows = int(os.getenv("MAX_ROWS", "0"))

claim_generation_prompt = f"""A **claim** is a statement or assertion made within a text that expresses a belief, opinion, or fact. Given the evidence and the original context, please transform the evidence into a claim.

Note:
- The claim should be a clear and concise statement that logically follows from the provided evidence.
- The claim should not contain ambiguous references such as "he," "she," or "it." Use complete names or specify entities where necessary.
- The claim must be a paraphrased version of the evidence, stating the point or fact clearly, without adding extra information.
- If there is no claim that can be drawn from the evidence, please leave the response blank."""

df = pd.read_csv(f"{folder_path}/sentences_isfact.csv")
df = df[df['is_fact']].rename(columns={'text': 'sentence'})

chunk_list = pd.read_csv(f"{folder_path}/chunk.csv")['body'].tolist()

df = df.drop_duplicates(subset=['sentence', 'chunk_id']).reset_index(drop=True)
df['chunk'] = df['chunk_id'].apply(lambda x: chunk_list[int(x)])

if max_rows > 0:
    df = df.head(max_rows)

itc, otc = 0, 0
claims = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    user_prompt = f"""Context: {row['chunk']}
Evidence: {row['sentence']}
Claim: """
    
    try:
        completion = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "assistant", "content": claim_generation_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=1000,
        )
        itc += completion.usage.prompt_tokens
        otc += completion.usage.completion_tokens
        response = completion.choices[0].message.content.strip()
        
        # Retry if empty response
        if len(response) < 2:
            print("empty response!")
            completion = client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "assistant", "content": claim_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=3000,
            )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"[{i}] Error: {e}")
        response = ""  # handle empty response

    claims.append({
        'claim': response.replace("**Claim:**", '').replace("Claim:", '').strip(),
        'sentence': row['sentence'],
        'chunk_id': row['chunk_id']
    })

    print(i, 1.1/1000000*itc + 4.4/1000000*otc)
    pd.DataFrame(claims).to_csv(f"{folder_path}/claim.csv", index=False)

claim_df = pd.DataFrame(claims)
claim_df.to_csv(f"{folder_path}/claim.csv", index=False)
print(i, 1.1/1000000*itc + 4.4/1000000*otc)