import pandas as pd
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# 누적 토큰 수 및 비용 초기화
total_itc = 0
total_otc = 0

# 일관성 판단 함수
def check_consistency(row):
    global total_itc, total_otc

    system_prompt = """You are an AI assistant that receives pairs of sentences and claims.
Your task is to determine whether each claim is consistent with its corresponding sentence.
Focus solely on whether the claim accurately reflects the core factual content of the sentence.
Ignore style, tone, attitude, or figurative language.
Respond with "Yes" if the claim is factually consistent with the sentence.
Respond with "No" if the claim introduces information that is not supported or is inconsistent.
Output format: Yes / No"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": f"Sentence: {row['sentence']}\nClaim: {row['claim']}"}
        ],
        temperature=0.0,
        max_tokens=5,
    )

    # 토큰 수 누적
    itc = completion.usage.prompt_tokens
    otc = completion.usage.completion_tokens
    total_itc += itc
    total_otc += otc

    # 응답 처리
    response = completion.choices[0].message.content.strip()
    result = 'yes' in response.lower()
    print(result)
    return result

folder_path = os.getenv("FOLDER_PATH", "demo")

# CSV 파일 읽기
df = pd.read_csv(f"{folder_path}/claim.csv").drop_duplicates(subset=['claim'])
max_rows = int(os.getenv("MAX_ROWS", "0"))
if max_rows > 0:
    df = df.head(max_rows)

# tqdm으로 처리 진행 표시
tqdm.pandas(desc="Checking consistency")
df['is_same'] = df.progress_apply(check_consistency, axis=1)

# 총 가격 계산
total_price = 0.15/1000000 * total_itc + 0.6/1000000 * total_otc
print(f"Total estimated cost: ${total_price}")

# 결과 저장
df.to_csv(f"{folder_path}/claim_issame.csv", index=False)