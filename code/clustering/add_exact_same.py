import pandas as pd
import os

folder_path = os.getenv("FOLDER_PATH", "demo")

# Load data
final_df = pd.read_csv(f"{folder_path}/same_entity_merged.csv")
final_df['entities'] = final_df['entities'].apply(eval)

always_df = final_df[final_df['type'] == "always"]
triple_df = pd.read_csv(f"{folder_path}/triple_contextually_same.csv")

# Step 1: Build synonym mapping dictionary
entity_map = {}
for _, row in always_df.iterrows():
    representative = row["entities"][0]  # representative entity
    for alias in row["entities"]:
        entity_map[alias] = representative  # map all aliases to representative

log_path = f"{folder_path}/entity_exact_same_log.txt"
changed_rows = set()  # 변경된 row index를 기록할 집합

# Open log file
with open(log_path, "w", encoding="utf-8") as log_file:
    def normalize_entity(entity, idx, col):
        normalized_entity = entity_map.get(entity, entity)
        if normalized_entity != entity:
            log_file.write(f"{entity} -> {normalized_entity}\n")
            changed_rows.add(idx)  # remember changed index
        return normalized_entity

    # Apply to source column
    triple_df["source"] = triple_df.apply(lambda row: normalize_entity(row["source"], row.name, "source"), axis=1)
    # Apply to target column
    triple_df["target"] = triple_df.apply(lambda row: normalize_entity(row["target"], row.name, "target"), axis=1)

# Remove rows where source equals target
print(len(triple_df))
triple_df = triple_df[triple_df["source"] != triple_df["target"]]
print(len(triple_df))

# Mark only changed rows as True (preserve existing True values)
triple_df["is_added"] = triple_df.apply(
    lambda row: True if row["is_added"] or row.name in changed_rows else False,
    axis=1
)

# Save results
triple_df.to_csv(f"{folder_path}/triple_exact_same.csv", index=False)