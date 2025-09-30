import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import re
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI()
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

rag_generation_prompt = """You are an AI assistant designed to generate answers for multi-hop questions. Given a question and its corresponding context, use only the information in the context to generate a **specific, concise answer**.

The answer should be **a clear, short entity, concept, or term**, such as "Microsoft", "United States", or "2020". Do not provide detailed explanations or longer sentences. 

Do not use any external knowledge or make assumptions. Focus solely on the information provided in the context to answer the question.

Output format:
Answer"""

folder_path = os.getenv("FOLDER_PATH", "demo")
model = os.getenv("LLM_MODEL", "gpt-4.1-mini")  # default via env

chunk_df = pd.read_csv(f"{folder_path}/DB_256.csv")
chunk_df['embedding'] = chunk_df['embedding'].apply(eval)

itc, otc = 0, 0
for hop in range(2,6):
    qa_path = f"{folder_path}/qa_{hop}hop_400sample_final.csv"
    if not os.path.exists(qa_path):
        continue
    print(f"Processing {hop}hop...")
    questions = pd.read_csv(qa_path)['question'].tolist()
    
    rag_result = []
    for question in tqdm(questions, total=len(questions)):
        question_embedding = embedder.embed_query(question)
        
        # cosine similarity
        embeddings_matrix = np.vstack(chunk_df['embedding'].to_numpy())
        sim_scores = cosine_similarity([question_embedding], embeddings_matrix)[0]
        
        # top-N contexts
        top_indices = np.argsort(sim_scores)[::-1][:10]
        top_contexts = chunk_df.iloc[top_indices]['chunk'].tolist()

        user_prompt = f"Question: {question}\n\nContext: {top_contexts}"
        
        if model == "o1-mini":
            completion = client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "assistant", "content": rag_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=3000,
            )
        elif model == "gpt-4o":
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": rag_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0,
            )
        elif model == "gpt-4o-mini":
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": rag_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0,
            )
        elif model == "gpt-4.1-mini":
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": rag_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=10,
                temperature=0,
            )
        itc += completion.usage.prompt_tokens
        otc += completion.usage.completion_tokens
        response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            if model == "o1-mini":
                completion = client.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "assistant", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=4000,
                )
            elif model == "gpt-4o":
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            elif model == "gpt-4o-mini":
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            elif model == "gpt-4.1-mini":
                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            if model == "o1-mini":
                completion = client.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "assistant", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=5000,
                )
            elif model == "gpt-4o":
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            elif model == "gpt-4o-mini":
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            elif model == "gpt-4.1-mini":
                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": rag_generation_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=10,
                    temperature=0,
                )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()
        rag_result.append({"rag_answer": response, "rag_context": top_contexts})

    print(1.1/1000000*itc + 4.4/1000000*otc)
    if model == "o1-mini":
        pd.DataFrame(rag_result).to_csv(f"{folder_path}/rag_{hop}hop_400sample_final_top10.csv", index=False)
    elif model == "gpt-4o":
        pd.DataFrame(rag_result).to_csv(f"{folder_path}/rag_{hop}hop_400sample_final_4o_top10.csv", index=False)
    elif model == "gpt-4o-mini":
        pd.DataFrame(rag_result).to_csv(f"{folder_path}/rag_{hop}hop_400sample_final_4o_mini_top10.csv", index=False)
    elif model == "gpt-4.1-mini":
        pd.DataFrame(rag_result).to_csv(f"{folder_path}/rag_{hop}hop_400sample_final_4.1_mini_top10.csv", index=False)