import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import re
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

load_dotenv()
client = OpenAI()
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# 토크나이저 로드
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
MAX_TOKENS = 8192

def truncate_to_token_limit(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def power_mean(values, p):
    values = np.array(values)
    if p == 0: # 기하평균(geometric mean)
        return np.exp(np.mean(np.log(values)))
    else:
        return (np.mean(values ** p)) ** (1 / p)

name = ['','','two','three','four','five']

folder_path = "health"
model = "o1-mini" # "o1-mini" or "gpt-4o" or "gpt-4o-mini"

means = []
# details = []
final = []
p_value = -3
for hop in range(2,6):
    print(hop,"hop", folder_path, model)

    qa = pd.read_csv(f"{folder_path}/qa_{hop}hop_400sample_final.csv")
    if model=="o1-mini":
        path = f"{folder_path}/llm_score_{hop}hop_400sample_top10.csv"
    elif model=="gpt-4o":
        path = f"{folder_path}/llm_score_{hop}hop_400sample_4o_top10.csv"
    elif model=="gpt-4o-mini":
        path = f"{folder_path}/llm_score_{hop}hop_400sample_4o_mini_top10.csv"
    else:
        print(f"Wrong model: {model}")
        break
    df = pd.concat([qa,pd.read_csv(path)], axis=1)

    df['chunks'] = df['chunks'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    result = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        question_text = truncate_to_token_limit(row['question'])
        question_embedding = embedder.embed_query(question_text)
        similarity_list = []
        
        for chunk_text in row['chunks']:
            chunk_text = truncate_to_token_limit(chunk_text)
            chunk_embedding = embedder.embed_query(chunk_text)
            sim = cosine_similarity(
                np.array(question_embedding).reshape(1, -1),
                np.array(chunk_embedding).reshape(1, -1)
            )[0][0]
            similarity_list.append(sim)
        
        # simple average
        # result.append(np.mean(similarity_list) if similarity_list else 0.0)
        
        # power mean 적용
        result.append(power_mean(similarity_list, p_value) if similarity_list else 0.0)

        # 최솟값
        # result.append(np.min(similarity_list) if similarity_list else 0.0)

    df['similarity'] = result

    ### 아래가 N개 라벨로 만든 경우###
    # 1. similarity를 4개 구간으로 분할 (0~1 범위)
    df['similarity_bin'] = pd.qcut(df['similarity'], q=4, labels=False)
    
    # 2. 각 구간별 평균 계산
    grouped = df.groupby('similarity_bin').agg({
        'similarity': 'mean',
        'is_correct': 'mean'
    }).reset_index()

    # 3. 구간 평균값으로 상관관계 계산
    correlation = grouped['similarity'].corr(grouped['is_correct'])
    print(f"Hop {hop} Correlation: {round(correlation,3)}")
    print("="*50)
    means.append(correlation)

    final.append(grouped['is_correct'].tolist())

reversed_final = [sublist[::-1] for sublist in final]
print("Matrix", p_value)
print(np.array(reversed_final))