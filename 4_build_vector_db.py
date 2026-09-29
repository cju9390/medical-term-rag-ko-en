import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("data/corpus/pubmed_diabetes.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pubmed_diabetes"


def load_corpus():
    docs = []
    with open(CORPUS_PATH, encoding="utf-8") as fp:
        for line in fp:
            docs.append(json.loads(line))
    return docs


def main():
    docs = load_corpus()
    print(f"코퍼스 로드: {len(docs)}건")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 제목 + 초록을 합쳐서 임베딩 (초록만 있으면 문맥이 부족한 경우가 있어 제목도 포함)
    texts = [f"{d['title']}\n{d['abstract']}" for d in docs]
    ids = [d["pmid"] for d in docs]
    metadatas = [
        {"title": d["title"], "mesh_terms": ", ".join(d["mesh_terms"])} for d in docs
    ]

    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # 재실행 시 중복 축적 방지를 위해 기존 컬렉션 초기화
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"벡터DB 저장 완료: {collection.count()}건 (collection={COLLECTION_NAME})")

    # 테스트 검색
    query = "type 2 diabetes mellitus management guidelines"
    query_embedding = embedder.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    print(f"\n테스트 검색 쿼리: {query}")
    for doc_id, meta, dist in zip(
        results["ids"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  - (PMID={doc_id}, 거리={dist:.4f}) {meta['title']}")


if __name__ == "__main__":
    main()
