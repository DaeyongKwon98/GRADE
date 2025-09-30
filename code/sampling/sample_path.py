import pandas as pd
import os

folder_path = os.getenv("FOLDER_PATH", "demo")

path = pd.read_csv(f"{folder_path}/DAG_path_shortest.csv").sort_values(by='hops', ascending=True)
grouped_counts = path.groupby('hops').size()
print(grouped_counts)
path = path[(path['hops']>=2) & (path['hops']<=5)]

path['first_entity'] = path['triples'].apply(lambda l: eval(l)[0][0])
path['last_entity'] = path['triples'].apply(lambda l: eval(l)[-1][-1])
path.drop_duplicates(subset=['first_entity', 'last_entity'], inplace=True)
grouped_counts = path.groupby('hops').size()
print(grouped_counts)

for i in range(2,6):
    one = path[path['hops']==i]
    # First sample up to 400
    if len(one) >= 400:
        one_sample = one.sample(n=400, random_state=42)
    else:
        one_sample = one.copy()

    # Optionally top up from remaining
    remaining = one[~one.index.isin(one_sample.index)]
    if len(one_sample) < 400 and len(remaining) > 0:
        need = min(400 - len(one_sample), len(remaining))
        one_sample = pd.concat([
            one_sample,
            remaining.sample(n=need, random_state=42)
        ], ignore_index=False)

    one_sample.to_csv(f"{folder_path}/{i}hop_qa_400sample.csv", index=False)