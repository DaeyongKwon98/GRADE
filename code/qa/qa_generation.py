import random
import pandas as pd
from tqdm import tqdm
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

load_dotenv()
client = OpenAI()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

qa_generation_prompt = """You are an AI assistant designed to generate multi-hop questions and answers based on triples in the form of (source_entity, relationship, target_entity), along with the associated claims and context.

Your task is to generate a multi-hop question-answer pair based on the given triples. The number of hops should correspond to the number of triples provided. If there are N triples, generate a question that connects all N triples, and use them to form a coherent, logical path for the answer.

Ensure that:

- The question should begin with "Question:" and the answer should begin with "Answer:".
- The question should clearly reference the entities and relationships, and should be designed such that the answer is a concise, **specific entity or short phrase** (e.g., "Microsoft", "United States", "2025", "GLP-1 drugs").
- The answer should **not be abstract** (e.g., "noticeable effects", "study participants", "potential limitations") but should be a **clear entity, specific term, or concise concept** that can be derived directly from the triples.
- The question and answer should be linked with a pipe (|) on the same line.
- Do not add external knowledge or assumptions beyond the given triples.

Notes for clarification:

- For N triples: The question should logically connect all N triples and form a coherent path that leads to a **specific, concrete answer** derived solely from the entities in the triples.
- Make sure the question is specific and each relationship in the chain is clearly traceable to lead to the final answer.

Example output format:
Question: Which company founded by Bill Gates owns the professional networking platform that was acquired in 2016?|Answer: Microsoft
Question: What type of medication, known to lower glucose levels, is prescribed by Dr. Brett Osborn?|Answer: GLP-1 drugs
Question: Which government entity implemented the loan program to support farmers affected by animal disease outbreaks in Minnesota?|Answer: Minnesota Department of Agriculture"""

def parse_qa(text):
    parts = text.split("|")

    if len(parts)!=2:
        print("Wrong response:", text)

    question = parts[0].replace("Question:", "").replace("question:", "").strip()
    answer = parts[1].replace("Answer:", "").replace("answer:", "").strip()
    return {'question': question, 'answer': answer}

folder_path = os.getenv("FOLDER_PATH", "demo")

itc, otc = 0, 0
inputs = []
for hop in range(2,6):
    path = f"{folder_path}/{hop}hop_qa_400sample.csv"
    if os.path.exists(path):
        inputs.append((hop, pd.read_csv(path)))
for hop, df in inputs:
    qa_result = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user_prompt = f"Triples: {row['triples']}\n\nClaims: {row['claims']}\n\nContext: {row['chunks']}"
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "assistant", "content": qa_generation_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=3000,
        )
        itc += completion.usage.prompt_tokens
        otc += completion.usage.completion_tokens
        response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "assistant", "content": qa_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=4000,
            )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()
        if len(response) < 2:
            print("empty response!")
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "assistant", "content": qa_generation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=5000,
            )
            itc += completion.usage.prompt_tokens
            otc += completion.usage.completion_tokens
            response = completion.choices[0].message.content.strip()
        
        for r in response.split('\n'):
            if "question:" in r.lower() and "answer:" in r.lower():
                qa_result.append(parse_qa(r))

    qa_df = pd.DataFrame(qa_result)
    if MAX_ROWS > 0:
        qa_df = qa_df.head(MAX_ROWS)
    print(1.1/1000000*itc + 4.4/1000000*otc)
    qa_df.to_csv(f"{folder_path}/qa_{hop}hop_400sample.csv", index=False)