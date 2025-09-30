import random
import pandas as pd
from tqdm import tqdm
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
client = OpenAI()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
folder_path = os.getenv("FOLDER_PATH", "demo")
max_rows = int(os.getenv("MAX_ROWS", "0"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

system_prompt = f"""You are an AI assistant that extracts entities and their relationships from a list of sentences.
Each sentence has an associated sentence ID.

Your task is to extract triplets from each sentence in the form of:
(source_entity|relationship|target_entity|sentence_id)

Please follow these guidelines:
- An entity can be a person, place, object, concept, or any meaningful noun phrase that participates in a relationship.
- Extract all valid (source_entity|relationship|target_entity) triplets from each sentence.
- Append the sentence ID at the end of each triplet to indicate which sentence it came from.
- If multiple triplets can be extracted from a single sentence, list all of them.
- Do not include duplicate triplets where only the order of source and target is reversed.

IMPORTANT: Resolve pronouns
- Replace pronouns such as he, she, it, they, this, that with the most specific entity mentioned in the sentence.

Output format:
(source_entity|relationship|target_entity|1)  
(source_entity|relationship|target_entity|2)  
(source_entity|relationship|target_entity|2)  
(source_entity|relationship|target_entity|3)"""

df = pd.read_csv(f"{folder_path}/claim_issame.csv")
df = df[df['is_same']]
print(len(df))
if max_rows > 0:
    df = df.head(max_rows)

batch_size = 10
itc, otc = 0, 0
dfs = []
for start in tqdm(range(0, len(df), batch_size)):
    end = start + batch_size
    batch = df.iloc[start:end]

    user_prompt = ""
    for i, sentence in enumerate(batch['claim']):
        user_prompt += f"Sentence {i+1}: {sentence}\n"
    # print(user_prompt)

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=4000,
    )
    itc += completion.usage.prompt_tokens
    otc += completion.usage.completion_tokens
    response = completion.choices[0].message.content.strip().strip('()')

    # print(response)

    l = response.split('\n')

    # triplet 추출
    pattern = r"\(([^|]+)\|([^|]+)\|([^|]+)\|(\d+)\)"
    matches = re.findall(pattern, response)

    # triplet -> DataFrame
    triplets = []
    for source, relation, target, sid in matches:
        sid_int = int(sid)
        if 1 <= sid_int <= len(batch):
            claim = batch.iloc[sid_int - 1]["claim"]
            triplets.append({
                "source": source.strip().lower(),
                "relationship": relation.strip().lower(),
                "target": target.strip().lower(),
                "claim": claim
            })

    triplet_df = pd.DataFrame(triplets)
    dfs.append(triplet_df)

price = 1.1/1000000*itc + 4.4/1000000*otc
print(price)
result_df = pd.concat(dfs)
result_df['is_added'] = False
result_df.to_csv(f"{folder_path}/triple.csv", index=False)
print(f"{len(result_df)} triplets extracted.")