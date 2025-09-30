import pandas as pd
import os
from collections import defaultdict
import itertools
import os

folder_path = os.getenv("FOLDER_PATH", "demo")

# 1. Load and preprocess synonym mappings
final_df = pd.read_csv(f"{folder_path}/same_entity_merged.csv")
final_df['entities'] = final_df['entities'].apply(eval)
context_df = final_df[final_df['type'] == 'context']

# Convert cluster IDs to list of strings
def normalize_cluster_ids(x):
    if isinstance(x, str):
        x = eval(x)  # handle list stored as string
    if not isinstance(x, list):
        return [str(x)]
    return [str(i) for i in x]

context_df['cluster_id'] = context_df['cluster_id'].apply(normalize_cluster_ids)

# 2. For each cluster, store synonym groups: cluster_id → [set(entity1, entity2, ...)]
cluster_to_synonym_groups = defaultdict(list)
for _, row in context_df.iterrows():
    entity_group = set(e.lower() for e in row['entities'])
    for cid in row['cluster_id']:
        cluster_to_synonym_groups[cid].append(entity_group)

# 3. triple 데이터 로딩 및 병합
clustered_df = pd.read_csv(f"{folder_path}/claim_clustered.csv")
triple_claim = pd.read_csv(f"{folder_path}/triple.csv")
triple_claim['claim'] = triple_claim['claim'].apply(lambda x: x.strip('"'))
triple_df = pd.merge(triple_claim, clustered_df, on='claim').dropna()
triple_df['cluster_id'] = triple_df['cluster_id'].astype(str)

print(f"Original triple count: {len(triple_df)}")

# 4. Expand triples per cluster using contextual synonym groups
expanded_rows = []
log_path = os.path.join(folder_path, "expanded_triples_log.txt")
count = 0

with open(log_path, "w", encoding="utf-8") as log_file:
    for cid, group_df in triple_df.groupby("cluster_id"):
        synonym_groups = cluster_to_synonym_groups.get(cid, [])
        if not synonym_groups:
            expanded_rows.extend(group_df.to_dict(orient='records'))
            continue

        for _, row in group_df.iterrows():
            src = row['source'].lower()
            tgt = row['target'].lower()

            # Find synonym groups
            src_group = next((g for g in synonym_groups if src in g), {src})
            tgt_group = next((g for g in synonym_groups if tgt in g), {tgt})

            for new_src, new_tgt in itertools.product(src_group, tgt_group):
                if new_src == new_tgt:
                    continue  # skip self

                new_row = row.to_dict()
                new_row['source'] = new_src
                new_row['target'] = new_tgt
                new_row['is_added'] = True
                expanded_rows.append(new_row)

                if new_src != src or new_tgt != tgt:
                    line = f"Expanded triple: ({src}, {row['relationship']}, {tgt}) → ({new_src}, {row['relationship']}, {new_tgt})\n"
                    log_file.write(line)
                    count += 1

# 5. Merge and save
expanded_df = pd.DataFrame(expanded_rows)

# Columns used for duplicate removal
dedup_cols = ["source", "relationship", "target", "claim", "sentence", "chunk_id", "cluster_id"]

# Group by dedup columns and set is_added True if any row is True
expanded_df = (
    expanded_df.groupby(dedup_cols, as_index=False)
    .agg({'is_added': lambda x: any(x)})
)

print(f"Triple count after dedup: {len(expanded_df)}")
expanded_df.to_csv(f"{folder_path}/triple_contextually_same.csv", index=False)