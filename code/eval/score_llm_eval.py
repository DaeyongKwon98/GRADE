import pandas as pd
from tqdm import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
folder_path = os.getenv("FOLDER_PATH", "demo")

llm_eval_prompt = """You are an AI assistant that receives a question along with two answers: a ground truth answer and a generated response. Your task is to evaluate whether the generated response is correct or not, and provide a binary judgment (True or False).
Output format:
True/False"""

model = os.getenv("LLM_MODEL", "gpt-4.1-mini")  # default via env

itc, otc = 0, 0
for hop in range(2,6):
    path_gt = f"{folder_path}/qa_{hop}hop_400sample_final.csv"
    if not (os.path.exists(path_gt) and os.path.getsize(path_gt) > 0):
        continue
    print(f"Processing {hop}hop...")
    try:
        gt = pd.read_csv(path_gt).fillna("")
    except Exception:
        continue
    if model == "o1-mini":
        rag = pd.read_csv(f"{folder_path}/rag_{hop}hop_400sample_final_top10.csv").fillna("")
    elif model == "gpt-4o":
        rag = pd.read_csv(f"{folder_path}/rag_{hop}hop_400sample_final_4o_top10.csv").fillna("")
    elif model == "gpt-4o-mini":
        rag = pd.read_csv(f"{folder_path}/rag_{hop}hop_400sample_final_4o_mini_top10.csv").fillna("")
    else:
        rag_path = f"{folder_path}/rag_{hop}hop_400sample_final_4.1_mini_top10.csv"
        if not (os.path.exists(rag_path) and os.path.getsize(rag_path) > 0):
            print(f"Skipping {hop}hop eval: rag file missing/empty")
            continue
        try:
            rag = pd.read_csv(rag_path).fillna("")
        except Exception:
            print(f"Skipping {hop}hop eval: rag file unreadable")
            continue
    df = pd.concat([gt,rag], axis=1)
    
    result = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = f"Question: {row['question']}\nGround Truth Answer: {row['answer']}\nResponse: {row['rag_answer']}"
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "assistant", "content": llm_eval_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=800,
        )
        itc += completion.usage.prompt_tokens
        otc += completion.usage.completion_tokens
        response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "assistant", "content": llm_eval_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1500,
            )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()
        if 'true' in response.lower():
            result.append({'is_correct': True})
        else:
            result.append({'is_correct': False})

    print(1.1/1000000*itc + 4.4/1000000*otc)
    if model == "o1-mini":
        pd.DataFrame(result).to_csv(f"{folder_path}/llm_score_{hop}hop_400sample_top10.csv", index=False)
    elif model == "gpt-4o":
        pd.DataFrame(result).to_csv(f"{folder_path}/llm_score_{hop}hop_400sample_4o_top10.csv", index=False)
    elif model == "gpt-4o-mini":
        pd.DataFrame(result).to_csv(f"{folder_path}/llm_score_{hop}hop_400sample_4o_mini_top10.csv", index=False)
    elif model == "gpt-4.1-mini":
        pd.DataFrame(result).to_csv(f"{folder_path}/llm_score_{hop}hop_400sample_4.1_mini_top10.csv", index=False)