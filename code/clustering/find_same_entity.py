import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()
client = OpenAI()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
folder_path = os.getenv("FOLDER_PATH", "demo")
MAX_GROUPS = int(os.getenv("MAX_GROUPS", "0"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


clustered_df = pd.read_csv(f"{folder_path}/claim_clustered.csv")
triple_claim = pd.read_csv(f"{folder_path}/triple.csv")
triple_claim['claim'] = triple_claim['claim'].apply(lambda x: x.strip('"'))

print(f"len triple: {len(triple_claim)}, len clustered: {len(clustered_df)}")

merged_df = pd.merge(triple_claim,clustered_df,on='claim')
print(f"len merged_df: {len(merged_df)}")

# 지정한 컬럼을 소문자로 변환
cols_to_lower = ['source', 'relationship', 'target', 'sentence', 'claim']

for col in cols_to_lower:
    merged_df[col] = merged_df[col].str.lower()

# merged_df에서 cluster_id를 기준으로 그룹화
grouped = merged_df.groupby('cluster_id')

find_same_entity_system_prompt = f"""You are an AI assistant tasked with identifying entities that refer to the same concept based on a given set of triples and their supporting claims.

Each input consists of multiple (source_entity, relationship, target_entity) triples along with their corresponding claim context.
Your task is to group entities that can be considered the same, based on both the triples and their claim contexts.

There are two types of equivalence:
1. Always equivalent: Entities that refer to the same real-world object or concept in any context (e.g., "USA" and "United States").
2. Context-dependent equivalent: Entities that refer to the same thing only in the context of the given triples and claim(s) (e.g., "study co-author" and "microplastics researcher").

Format your output as follows:
Group identical entities together inside square brackets [].
Separate each entity with a vertical bar |.
At the end of each group, append either "always" or "context" (in quotes) to indicate the type of equivalence.
Write one group per line.
If no identical entities are found, output exactly: No identical entities found.

Example output:
[USA|United States|"always"]
[Tesla, Inc.|Tesla Motors|Tesla|"always"]
[microplastics researcher|study co-author|"context"]"""

# 각 그룹에 대해 처리
itc, otc = 0, 0
entity_groups_by_cluster_id = {}

processed_groups = 0
for cluster_id, group in tqdm(grouped):
    print(f"Processing cluster_id: {cluster_id}")

    used_entities = set(group['source'].tolist() + group['target'].tolist())
    print(len(used_entities), used_entities)
    
    user_prompt = ""
    # claim 기준으로 triple 묶기
    claim_grouped = group.groupby(['sentence', 'claim'])

    for idx, ((sentence, claim), sub_group) in enumerate(claim_grouped):
        for _, row in sub_group.iterrows():
            user_prompt += f"Triple: ({row['source']}, {row['relationship']}, {row['target']})\n"
        user_prompt += f"Claim: {claim}\n\n"

    # print(user_prompt)

    # 첫 요청
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "assistant", "content": find_same_entity_system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=3000,
    )
    itc += completion.usage.prompt_tokens
    otc += completion.usage.completion_tokens
    response = completion.choices[0].message.content.strip()

    # 응답이 비어있으면 재요청
    if len(response) < 2:
        print("empty response! retrying...")
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "assistant", "content": find_same_entity_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=4000,
        )
        itc += completion.usage.prompt_tokens
        otc += completion.usage.completion_tokens
        response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("still empty response..")

    # 엔티티 그룹 파싱 및 클렌징 (모든 entity를 소문자로 처리)
    entity_groups = []
    for line in response.strip().split("\n"):
        if line.strip() and line.startswith("[") and line.endswith("]"):
            # [] 제거 후 split
            entities_and_type = line.strip()[1:-1].split("|")
            # 마지막 요소가 'always' 혹은 'context'인 경우
            if len(entities_and_type) > 1:
                entities = [ent.strip().lower().replace('\u202f', ' ') for ent in entities_and_type[:-1]]
                entities = [e for e in entities if e in used_entities]
                entity_type = entities_and_type[-1].strip().strip('"').lower()
                if len(set(entities)) > 1:
                    entity_groups.append({"entities": entities, "type": entity_type})
            else:
                entities = [ent.strip().lower().replace('\u202f', ' ') for ent in entities_and_type]
                entities = [e for e in entities if e in used_entities]
                if len(set(entities)) > 1:
                    entity_groups.append({"entities": entities, "type": "context"}) # 기본적으로 context로 처리

    entity_groups_by_cluster_id[cluster_id] = entity_groups
    processed_groups += 1
    if MAX_GROUPS > 0 and processed_groups >= MAX_GROUPS:
        break

    # JSON 저장
    with open(f"{folder_path}/entity_groups_by_cluster_id.json", "w", encoding="utf-8") as f:
        json.dump(entity_groups_by_cluster_id, f, ensure_ascii=False, indent=2)

print(1.1/1000000*itc + 4.4/1000000*otc)

####
# 각 클러스터를 개별 데이터프레임으로 만들고 cluster_id 추가
dfs = []

for cluster_id, groups in entity_groups_by_cluster_id.items():
    if not groups:  # 빈 리스트는 건너뜀
        continue
    df = pd.DataFrame(groups)
    df['cluster_id'] = int(cluster_id)
    dfs.append(df)

# 모든 데이터프레임을 하나로 합치기
final_df = pd.concat(dfs, ignore_index=True)
final_df['entities'] = final_df['entities'].apply(lambda l: list(set([x.lower() for x in l])))
final_df = final_df[final_df['entities'].apply(lambda x: len(x)>1)].reset_index(drop=True)
final_df.to_csv(f"{folder_path}/same_entity.csv", index=False)

# 한번 더 중복 검사 수행 -> 최종 데이터 확정
from collections import defaultdict

# 유니온-파인드 정의
class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

# 1. 유니온-파인드 준비
uf = UnionFind()
entity_to_rows = defaultdict(list)

# 2. 항상 병합되는 그룹만 Union
for idx, row in final_df.iterrows():
    if row['type'] != 'always':
        continue  # context 타입은 제외
    entities = [e.lower() for e in row['entities']]
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            uf.union(entities[i], entities[j])
    for ent in entities:
        entity_to_rows[uf.find(ent)].append(idx)

# 3. 루트 기준으로 그룹핑
group_map = defaultdict(set)
for ent, rows in entity_to_rows.items():
    root = uf.find(ent)
    group_map[root].update(rows)

# 4. 병합된 always 그룹 생성
merged_data = []
seen = set()

for group_indices in group_map.values():
    merged_row = {
        'entities': set(),
        'type': 'always',
        'cluster_id': set()
    }
    for idx in group_indices:
        if idx in seen:
            continue
        seen.add(idx)
        row = final_df.iloc[idx]
        merged_row['entities'].update([e.lower() for e in row['entities']])
        merged_row['cluster_id'].add(int(row['cluster_id']))
    if len(merged_row['entities']) > 1:
        merged_data.append({
            'entities': sorted(merged_row['entities']),
            'type': 'always',
            'cluster_id': sorted(merged_row['cluster_id'])
        })

# 5. context 타입은 병합 없이 개별로 유지
for idx, row in final_df.iterrows():
    if row['type'] == 'context' and idx not in seen:
        merged_data.append({
            'entities': sorted([e.lower() for e in row['entities']]),
            'type': 'context',
            'cluster_id': [int(row['cluster_id'])]
        })

# 6. 최종 결과
final_df = pd.DataFrame(merged_data)
final_df.to_csv(f"{folder_path}/same_entity_merged.csv", index=False)
