import importlib.util
import json
from pathlib import Path

EVAL_SET_PATH = Path("data/eval/eval_set.jsonl")


def _load_generate_answer_module():
    path = Path(__file__).parent / "5_generate_answer.py"
    spec = importlib.util.spec_from_file_location("generate_answer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_set():
    with open(EVAL_SET_PATH, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def main(k=3):
    eval_items = load_eval_set()
    ga = _load_generate_answer_module()

    term_lookup = ga._load_term_lookup_module()
    mapper = term_lookup.TermMapper(term_lookup.load_dictionary())

    hits = 0
    results = []
    for item in eval_items:
        # 평가셋 정답은 PubMed(영문 코퍼스) 문서라서 영문 검색 결과만 채점 대상으로 사용
        terms = mapper.extract_terms(item["question_ko"])
        en_retrieved = ga.retrieve_en(item["question_ko"], terms, n_results=k)
        retrieved_pmids = [r["id"] for r in en_retrieved]
        hit = item["expected_pmid"] in retrieved_pmids
        hits += hit
        results.append(
            {
                "id": item["id"],
                "question_ko": item["question_ko"],
                "expected_pmid": item["expected_pmid"],
                "retrieved_pmids": retrieved_pmids,
                "hit": hit,
            }
        )

    print("\n" + "=" * 60)
    print(f"Recall@{k}: {hits}/{len(eval_items)} = {hits / len(eval_items):.1%}")
    print("=" * 60)
    for r in results:
        mark = "O" if r["hit"] else "X"
        print(f"[{mark}] #{r['id']} {r['question_ko']}")
        if not r["hit"]:
            print(f"      기대: {r['expected_pmid']}  실제검색: {r['retrieved_pmids']}")

    return results


if __name__ == "__main__":
    main()
