import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
import os

# Load triple data
folder_path = os.getenv("FOLDER_PATH", "demo")
df = pd.read_csv(f"{folder_path}/triple_exact_same.csv")

# Create directed graph with relationship as edge attribute
print("Creating graph...")
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['source'], row['target'], relationship=row['relationship'])

all_paths = dict()  # key: (src, tgt), value: list of paths (triples)
node_list = list(G.nodes)

print("Finding all shortest paths only...")
for start_node in tqdm(node_list, desc="Start nodes"):
    for end_node in node_list:
        if start_node != end_node and nx.has_path(G, start_node, end_node):
            try:
                for path in nx.all_shortest_paths(G, start_node, end_node):
                    triple_path = []
                    for i in range(len(path) - 1):
                        src = path[i]
                        tgt = path[i + 1]
                        rel = G[src][tgt]['relationship']
                        triple_path.append({'source': src, 'relationship': rel, 'target': tgt})
                    key = (start_node, end_node)
                    all_paths.setdefault(key, []).append(triple_path)
            except nx.NetworkXNoPath:
                continue

shortest_paths = []
for paths in all_paths.values():
    shortest_paths.extend(paths)

# Load chunk data
chunk_df = pd.read_csv(f"{folder_path}/chunk.csv")
print(len(chunk_df),"chunks")

# Prepare triple lookup from original df
print("Preparing triple lookup...")
triple_lookup = {}
for _, row in df.iterrows():
    key = (row['source'], row['relationship'], row['target'])
    triple_lookup.setdefault(key, []).append({
        'claim': row['claim'],
        'chunk_id': row['chunk_id']
    })

# Map chunk_id to body text
chunk_id_to_body = dict(enumerate(chunk_df['body']))

# Compile results
print("Compiling results...")
results = []
for path in tqdm(shortest_paths):
    if len(path) >= 6:  # ✅ 6 hop 이상 경로 제외
        continue
    record = {
        'triples': [],
        'claims': [],
        'chunks': []
    }
    seen_claims = set()
    seen_chunks = set()
    
    for triple in path:
        src, rel, tgt = triple['source'], triple['relationship'], triple['target']
        record['triples'].append((src, rel, tgt))
        key = (src, rel, tgt)
        if key in triple_lookup:
            for info in triple_lookup[key]:
                if info['claim'] not in seen_claims:
                    record['claims'].append(info['claim'])
                    seen_claims.add(info['claim'])
                if info['chunk_id'] not in seen_chunks:
                    body = chunk_id_to_body.get(info['chunk_id'])
                    if body:
                        record['chunks'].append(body)
                        seen_chunks.add(info['chunk_id'])

    results.append(record)

# Convert to DataFrame
r = pd.DataFrame(results)
r['triples'] = r['triples'].apply(json.dumps)
r['claims'] = r['claims'].apply(json.dumps)
r['chunks'] = r['chunks'].apply(json.dumps)
r['hops'] = r['triples'].apply(lambda x: len(json.loads(x)))
r['claim_count'] = r['claims'].apply(lambda x: len(json.loads(x)))
r['chunk_count'] = r['chunks'].apply(lambda x: len(json.loads(x)))

# Save to CSV
output_path = f"{folder_path}/DAG_path_shortest.csv"
r.to_csv(output_path, index=False)
print(f"{len(r)} paths are created.")