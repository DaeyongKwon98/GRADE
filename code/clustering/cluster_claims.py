import logging
import random
from typing import List, Optional
import umap
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import os
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedder = OpenAIEmbeddings(model="text-embedding-3-small")
folder_path = os.getenv("FOLDER_PATH", "demo")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 아래 5개 함수는 cluster_embeddings를 위한 하위 함수임.
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings

def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings

def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int=0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters

def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters

def cluster_embeddings(df, threshold=0.1, dim=10, verbose=False):
    # Extract embeddings from merged_df
    embeddings = np.array(df['embedding'].tolist())

    # Perform clustering on the embeddings using the `perform_clustering` method
    clusters = perform_clustering(
        embeddings, dim=dim, threshold=threshold, verbose=verbose
    )

    # Initialize a new list to hold cluster assignments for each question
    cluster_labels = []

    # Flatten the cluster list into a format where each embedding gets assigned a list of clusters
    for cluster in clusters:
        cluster_labels.append(list(set(cluster)))

    # Add clustering results back into the dataframe
    df['clusters'] = cluster_labels
    return df
#########

def cluster_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    claim 임베딩을 기반으로 클러스터링하고,
    원본 df에 cluster_id를 추가한 후 반환.

    Args:
        df (pd.DataFrame): 'claim', 'sentence', 'chunk_id' 컬럼을 가진 데이터프레임

    Returns:
        pd.DataFrame: 'claim', 'sentence', 'chunk_id', 'cluster_id' 컬럼 포함 데이터프레임
    """
    # 1. claim을 임베딩
    df['embedding'] = df['claim'].apply(lambda x: embedder.embed_query(x))

    # 2. 클러스터링
    df = cluster_embeddings(df, threshold=0.1, dim=10, verbose=True)

    # 3. 각 row마다 가장 첫 번째 cluster id를 할당 (clusters는 리스트 형태임)
    df['cluster_id'] = df['clusters'].apply(lambda x: int(x[0]) if len(x) > 0 else -1)

    # 4. 필요한 컬럼만 반환
    clustered_df = df[['claim', 'sentence', 'chunk_id', 'cluster_id']].copy()

    return clustered_df

folder_path = os.getenv("FOLDER_PATH", "demo")

claim_df = pd.read_csv(f"{folder_path}/claim_issame.csv")
claim_df = claim_df[claim_df['is_same']].drop_duplicates(subset=['claim'])

triple_claim_list = pd.read_csv(f"{folder_path}/triple.csv")['claim'].tolist()

print(len(claim_df))

claim_df = claim_df[claim_df['claim'].isin(triple_claim_list)]

print(len(claim_df))

clustered_df = cluster_sentences(claim_df)
clustered_df.to_csv(f"{folder_path}/claim_clustered.csv", index=False)