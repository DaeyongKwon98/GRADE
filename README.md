# EMNLP_GRADE_MultiHop_QA_Generation

This repository accompanies the paper GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation (Accepted to EMNLP Findings 2025). See the paper: [arXiv PDF](https://arxiv.org/pdf/2508.16994).

## Environment setup
- Conda (original environment list):
```
conda create --name <env> --file requirements.txt
```
- Or lightweight pip env:
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-pip.txt
```

## Environment variables (.env)
Create a `.env` in the project root (see `.env.example`):
```
OPENAI_API_KEY=...
MEDIASTACK_API_KEY=...
LLM_MODEL=gpt-4.1-mini
```
All scripts automatically load `.env` via `python-dotenv`.

## License
This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## Project structure
```
code/
  ingestion/
    crawl_news.py
  preprocessing/
    is_fact.py
  claims/
    generate_claim_from_sentence.py
    claim_sentence_consistency.py
  triples/
    extract_triples.py
  clustering/
    cluster_claims.py
    find_same_entity.py
    add_contextually_same_path.py
    add_exact_same.py
  graph/
    find_DAG_path_shortest.py
  sampling/
    sample_path.py
  qa/
    qa_generation.py
    final_validation.py
  db/
    generate_db_from_chunk.py
  rag/
    rag_answer.py
  eval/
    score_llm_eval.py
    retriever_correlation.py
```

## Pipeline (execution order)
1. `code/ingestion/crawl_news.py`
   - Collect news via mediastack API; keep 512–8192-token bodies
2. `code/preprocessing/is_fact.py`
   - Sentence-split and filter factual sentences using `lighteternal/fact-or-opinion-xlmr-el`
3. `code/claims/generate_claim_from_sentence.py`
   - Generate claims from sentence and chunk
4. `code/claims/claim_sentence_consistency.py`
   - Check claim–sentence consistency
5. `code/triples/extract_triples.py`
   - Extract triples from claims (batched)
6. `code/clustering/cluster_claims.py`
   - Soft clustering of claims (GMM)
7. `code/clustering/find_same_entity.py`
   - Identify exact/contextually same entities within clusters
8. `code/clustering/add_contextually_same_path.py`
   - Add paths via contextual equivalence (A→C, D→A implies B→C, D→B if A≈B)
9. `code/clustering/add_exact_same.py`
   - Normalize entities by exact equivalence (unify to representative)
10. `code/graph/find_DAG_path_shortest.py`
   - Compute shortest paths on the DAG (skip start=end duplicates)
11. `code/sampling/sample_path.py`
   - Sample 400 paths per hop after dedup by start/end entities
12. `code/qa/qa_generation.py`
   - Generate ground-truth QA
13. `code/qa/final_validation.py`
   - Remove ambiguous QA (decontextualization check)
14. `code/db/generate_db_from_chunk.py`
   - Split chunks (min 128, max 256 tokens, overlap 50)
15. `code/rag/rag_answer.py`
   - Generate RAG answers (Top-10 chunks)
16. `code/eval/score_llm_eval.py`
   - Measure generator-side performance per hop
17. `code/eval/retriever_correlation.py`
   - Correlate query–chunk similarity with performance; build 4×4 matrix (by hop and similarity bin)

## Quick start
Set your domain folder name once via env var (used across steps):
```
export FOLDER_PATH=demo  # e.g., business | science | technology | health | entertainment
```
```
cd code
python ingestion/crawl_news.py    # requires MEDIASTACK_API_KEY
python preprocessing/is_fact.py
python claims/generate_claim_from_sentence.py
python triples/extract_triples.py
python clustering/cluster_claims.py
python clustering/find_same_entity.py
python clustering/add_contextually_same_path.py
python clustering/add_exact_same.py
python graph/find_DAG_path_shortest.py
python sampling/sample_path.py
python qa/qa_generation.py
python qa/final_validation.py
python db/generate_db_from_chunk.py
python rag/rag_answer.py
python eval/score_llm_eval.py
python eval/retriever_correlation.py
```

## Demo results (Sports)
- Crawl size: ~500 articles (`FOLDER_PATH=sportsdemo`)
- Sampled paths (post-dedup): 2-hop 110, 3-hop 28, 4-hop 1
- Final QA (after validation):
  - 2-hop: 72 items → accuracy 0.764 (LLM eval)
  - 3-hop: 13 items → accuracy 0.538 (LLM eval)

## 실행 예시
- 뉴스 수집
```
cd code
python crawl_news.py  # .env의 MEDIASTACK_API_KEY 필요
```

- DB 생성 (256 토큰 청크, overlap 50)
```
python generate_db_from_chunk.py
```

- QA 생성 및 RAG 평가 (OpenAI API 필요)
```
python qa.py
python rag.py
python score_llm_eval.py
python retriever_correlation.py
```

## 공개 릴리즈 노트
- 하드코딩된 API 키 제거 → `.env` 기반으로 로드
- `OPENAI_API_KEY`, `MEDIASTACK_API_KEY` 사용
- 재현 가능한 실행을 위해 사용 스크립트별 입력/출력 경로 주석 보강
