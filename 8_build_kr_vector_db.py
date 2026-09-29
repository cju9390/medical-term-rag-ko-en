import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path("data/corpus/kdca_diabetes_kr.jsonl")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "kdca_diabetes_kr"
# 한글 문서를 다국어로 임베딩하기 위한 모델 (영문 코퍼스용 all-MiniLM-L6-v2와는 별도)
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def chunk_text(text, max_len=500):
    """긴 문서를 문단 단위로 묶어 max_len 글자 내외 청크로 분할"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 1 > max_len:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n{p}".strip() if current else p
    if current:
        chunks.append(current)
    return chunks


def main():
    docs = load_corpus()
    print(f"한글 코퍼스 로드: {len(docs)}건")

    chunk_records = []
    for d in docs:
        for i, chunk in enumerate(chunk_text(d["content"])):
            chunk_records.append(
                {
                    "id": f"{d['cntnts_sn']}_{i}",
                    "text": chunk,
                    "title": d["title"],
                    "url": d["url"],
                    "license": d["license"],
                }
            )
    print(f"청크 생성: {len(chunk_records)}개 (평균 {sum(len(c['text']) for c in chunk_records)//len(chunk_records)}자)")

    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunk_records]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=16)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[c["id"] for c in chunk_records],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {"title": c["title"], "url": c["url"], "license": c["license"]}
            for c in chunk_records
        ],
    )
    print(f"벡터DB 저장 완료: {collection.count()}건 (collection={COLLECTION_NAME})")

    # 테스트 검색 (한글 질의 그대로, 번역 없이 직접 검색)
    query = "당뇨병 환자 운동할 때 주의할 점은?"
    query_embedding = embedder.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
    print(f"\n테스트 검색 쿼리: {query}")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  - (거리={dist:.4f}, 출처={meta['title']}) {doc[:80]}...")


if __name__ == "__main__":
    main()
