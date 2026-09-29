import csv
import difflib
from pathlib import Path

TERM_DICT_CSV = Path("data/kr_terminology/term_dictionary.csv")


def load_dictionary():
    with open(TERM_DICT_CSV, encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


class TermMapper:
    def __init__(self, terms):
        self.exact = {t["ko"]: t for t in terms}
        # 긴 용어부터 매칭해야 "당뇨병" 안의 "당" 같은 부분 오매칭을 피할 수 있음
        # 문장 내 추출(extract_terms)에는 1글자 용어를 빼서 "대사" 안의 "사"→Death 같은
        # 스퓨리어스 매칭을 막음 (단일 용어 조회는 lookup_exact로 여전히 가능)
        self.sorted_ko = sorted(
            (k for k in self.exact if len(k) >= 2), key=len, reverse=True
        )

    def lookup_exact(self, term):
        return self.exact.get(term)

    def lookup_fuzzy(self, term, cutoff=0.75, n=3):
        candidates = difflib.get_close_matches(term, self.exact.keys(), n=n, cutoff=cutoff)
        return [self.exact[c] for c in candidates]

    def map_term(self, term):
        hit = self.lookup_exact(term)
        if hit:
            return {"match_type": "exact", **hit}
        fuzzy = self.lookup_fuzzy(term)
        if fuzzy:
            return {"match_type": "fuzzy", **fuzzy[0]}
        return None

    def extract_terms(self, query):
        """문장 안에서 사전에 등재된 용어를 모두 찾아 매핑 (긴 용어 우선, 겹치는 구간은 재사용 안 함)"""
        found = []
        covered = [False] * len(query)
        for ko in self.sorted_ko:
            start = 0
            while True:
                idx = query.find(ko, start)
                if idx == -1:
                    break
                span = range(idx, idx + len(ko))
                if not any(covered[i] for i in span):
                    for i in span:
                        covered[i] = True
                    found.append({"start": idx, "end": idx + len(ko), **self.exact[ko]})
                start = idx + 1
        found.sort(key=lambda x: x["start"])
        return found


def main():
    terms = load_dictionary()
    mapper = TermMapper(terms)
    print(f"사전 로드 완료: {len(terms)}건\n")

    print("=== 1) 단일 용어 조회 (정확/유사매칭) ===")
    for q in ["당뇨병", "당뇨", "고혈압", "골절", "위암", "타이레놀"]:
        result = mapper.map_term(q)
        print(f"  {q:10s} -> {result}")

    print("\n=== 2) 문장에서 용어 추출 ===")
    sentence = "당뇨병 환자는 고혈압과 신장질환을 동반하는 경우가 많다."
    extracted = mapper.extract_terms(sentence)
    print(f"문장: {sentence}")
    for e in extracted:
        print(
            f"  - {e['ko']} -> {e['en']}  (code={e['code'] or '-'}, "
            f"umls={e['umls'] or '-'}, snomed={e['snomed'] or '-'}, source={e['source']})"
        )


if __name__ == "__main__":
    main()
