from langchain_text_splitters import TokenTextSplitter
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import re

tqdm.pandas()  # tqdm의 pandas 연동 활성화

load_dotenv()
client = OpenAI()
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

from tiktoken import encoding_for_model

# tiktoken 기반 토크나이저 사용
tokenizer = encoding_for_model("text-embedding-3-small")
token_splitter = TokenTextSplitter.from_tiktoken_encoder(
    chunk_size=256,
    chunk_overlap=50,
    model_name="text-embedding-3-small"  # splitter와 encoder 일치
)

folder_path = os.getenv("FOLDER_PATH", "demo")
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "0"))

chunk_df = pd.read_csv(f"{folder_path}/chunk.csv")

database = []
for article in tqdm(chunk_df['body']):
    chunks = token_splitter.split_text(article)
    
    min_tokens = 128
    final_chunks = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            chunk = buffer + " " + chunk
            buffer = ""
        if len(tokenizer.encode(chunk)) < min_tokens:
            buffer = chunk
        else:
            final_chunks.append({"chunk": chunk, "token_count": len(tokenizer.encode(chunk))})

    if buffer:
        final_chunks.append({"chunk": buffer, "token_count": len(tokenizer.encode(buffer))})
    
    database.extend(final_chunks)

db_df = pd.DataFrame(database)
if MAX_CHUNKS > 0:
    db_df = db_df.head(MAX_CHUNKS)
# db_df['embedding'] = db_df['chunk'].apply(lambda x: embedder.embed_query(x))
db_df['embedding'] = db_df['chunk'].progress_apply(lambda x: embedder.embed_query(x))
db_df.to_csv(f"{folder_path}/DB_256.csv", index=False)
print(f"{len(chunk_df)} articles become {len(db_df)} chunks.")