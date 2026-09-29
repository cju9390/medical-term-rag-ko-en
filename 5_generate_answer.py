import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
EN_COLLECTION = "pubmed_diabetes"
KR_COLLECTION = "kdca_diabetes_kr"
EN_EMBED_MODEL = "all-MiniLM-L6-v2"
KR_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma2:9b"


def _load_term_lookup_module():
    """2_term_lookup.py는 숫자로 시작해 일반 import가 안 되므로 파일 경로로 직접 로드"""
    path = Path(__file__).parent / "2_term_lookup.py"
    spec = importlib.util.spec_from_file_location("term_lookup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_ollama_running():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, ConnectionError):
        return False


def ollama_chat(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["message"]["content"].strip()


def build_search_query(query_ko, extracted):
    """사전 매핑은 힌트로만 주고, 최종 영문 검색어는 LLM이 번역/보완하도록 함
    (사전에 없는 약물명·해부학 용어 등도 LLM 번역으로 커버하기 위함)
    """
    hints = ", ".join(f"{e['ko']}={e['en']}" for e in extracted) or "(없음)"

    if not is_ollama_running():
        # Ollama 없으면 기존 방식(사전 매핑 단순 결합)으로 폴백
        return " ".join(e["en"] for e in extracted) if extracted else query_ko

    system_prompt = (
        "당신은 한글 의학 질문을 PubMed 검색에 적합한 영문 키워드로 번역하는 도구입니다. "
        "아래 '알려진 용어 매핑'을 참고하되, 매핑에 없는 용어도 의학 지식을 활용해 정확한 영문 용어로 번역하세요. "
        "설명 없이 검색에 쓸 영문 키워드만 출력하세요."
    )
    user_prompt = f"알려진 용어 매핑: {hints}\n질문: {query_ko}"
    try:
        return ollama_chat(system_prompt, user_prompt)
    except (urllib.error.URLError, ConnectionError):
        return " ".join(e["en"] for e in extracted) if extracted else query_ko


def retrieve_en(query_ko, extracted, n_results=3):
    """영문 PubMed 코퍼스: 사전 매핑(힌트) + LLM 번역 검색어로 검색"""
    search_query_en = build_search_query(query_ko, extracted)
    print(f"추출된 용어(힌트): {[(e['ko'], e['en']) for e in extracted]}")
    print(f"영문 검색 쿼리(LLM 번역): {search_query_en}")

    embedder = SentenceTransformer(EN_EMBED_MODEL)
    client_db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client_db.get_collection(EN_COLLECTION)

    query_embedding = embedder.encode([search_query_en])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)

    return [
        {
            "source_type": "pubmed",
            "id": pmid,
            "title": meta["title"],
            "text": doc,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "license": None,
        }
        for pmid, meta, doc in zip(
            results["ids"][0], results["metadatas"][0], results["documents"][0]
        )
    ]


def retrieve_kr(query_ko, n_results=3):
    """한글 KDCA 코퍼스: 번역 없이 다국어 임베딩으로 직접 검색"""
    embedder = SentenceTransformer(KR_EMBED_MODEL)
    client_db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client_db.get_collection(KR_COLLECTION)

    query_embedding = embedder.encode([query_ko])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)

    return [
        {
            "source_type": "kdca_kr",
            "id": chunk_id,
            "title": meta["title"],
            "text": doc,
            "url": meta["url"],
            "license": meta["license"],
        }
        for chunk_id, meta, doc in zip(
            results["ids"][0], results["metadatas"][0], results["documents"][0]
        )
    ]


def retrieve(query_ko, n_results=3):
    """한글 질의 -> 영문(PubMed)+한글(KDCA) 코퍼스 동시 검색 -> 결과 병합"""
    term_lookup = _load_term_lookup_module()
    terms = term_lookup.load_dictionary()
    mapper = term_lookup.TermMapper(terms)
    extracted = mapper.extract_terms(query_ko)

    en_results = retrieve_en(query_ko, extracted, n_results)
    kr_results = retrieve_kr(query_ko, n_results)
    return en_results + kr_results


def generate(query_ko, retrieved):
    context_parts = []
    for r in retrieved:
        if r["source_type"] == "pubmed":
            context_parts.append(f"[PMID:{r['id']}] {r['title']}\n{r['text']}")
        else:
            context_parts.append(
                f"[출처: {r['title']}, {r['url']} ({r['license']})]\n{r['text']}"
            )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "당신은 의학 문헌(영문 논문 초록 및 한글 건강정보)을 근거로 한글로 답변하는 보조 도구입니다. "
        "반드시 아래 context 안의 내용만 근거로 답변하고, context에 없는 내용은 모른다고 답하세요. "
        "답변은 한글로 작성하고, 끝에 참고한 출처(PMID 또는 출처명)를 표시하세요. "
        "이 답변이 진단/처방을 대체하지 않으며 전문가 확인이 필요하다는 점을 반드시 덧붙이세요."
    )
    user_prompt = f"Context:\n{context}\n\n질문: {query_ko}"
    return ollama_chat(system_prompt, user_prompt)


def main():
    query = "제2형 당뇨병 환자의 저혈당 위험 요인은 무엇인가요?"
    print("질문:", query)
    print("-" * 50)

    retrieved = retrieve(query)
    print("\n검색된 문서:")
    for r in retrieved:
        print(f"  - [{r['source_type']}:{r['id']}] {r['title']}")

    if not is_ollama_running():
        print("\nOllama 서버(localhost:11434)에 연결할 수 없어 답변 생성은 건너뜁니다. (검색까지는 정상 동작 확인)")
        return

    print("\n답변:")
    print(generate(query, retrieved))


if __name__ == "__main__":
    main()
