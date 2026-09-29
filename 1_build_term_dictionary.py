import csv
import json
import zipfile
from pathlib import Path

import openpyxl

AIHUB_DIR = Path(
    "308.AI 허브 데이터 활용을 위한 기계 번역앱 구축과 번역기 평가 및 신규 말뭉치 구축"
    "/01-1.정식개방데이터"
)
KCD_CSV = Path("건강보험심사평가원_상병마스터_20250930.csv")
KOSTOM_XLSX = Path("보건의료용어표준_V7.0_분야전체.xlsx")
OUT_CSV = Path("data/kr_terminology/term_dictionary.csv")
FIELDNAMES = ["ko", "en", "code", "umls", "snomed", "domain", "source"]


def load_kcd_terms():
    """상병마스터: 상병기호(KCD코드) + 한글명 + 영문명 -> 질병명 용어쌍"""
    terms = []
    with open(KCD_CSV, encoding="cp949", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ko = row["한글명"].strip()
            en = row["영문명"].strip()
            if ko and en:
                terms.append(
                    {
                        "ko": ko,
                        "en": en,
                        "code": row["상병기호"].strip(),
                        "umls": "",
                        "snomed": "",
                        "domain": "질병명",
                        "source": "KCD상병마스터",
                    }
                )
    return terms


def load_kostom_terms():
    """보건의료용어표준(KOSTOM): KCD 코드가 매핑된 진단 관련 행만 사용
    (전체 33.9만 행 중 검사/LOINC 등 비진단 항목을 빼고 9.6만 건으로 좁힘)
    """
    terms = []
    wb = openpyxl.load_workbook(KOSTOM_XLSX, read_only=True)
    ws = wb["V7.0"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        en, ko, umls, kcd, snomed = row[2], row[3], row[4], row[5], row[12]
        if ko and en and kcd:
            terms.append(
                {
                    "ko": str(ko).strip(),
                    "en": str(en).strip(),
                    "code": str(kcd).strip(),
                    "umls": str(umls).strip() if umls else "",
                    "snomed": str(snomed).strip() if snomed else "",
                    "domain": "질병명(KOSTOM)",
                    "source": "KOSTOM_V7.0",
                }
            )
    return terms


def load_aihub_medical_terms():
    """AI Hub 용어사전 zip(Training+Validation 라벨링데이터) 중 domain=='의학'만 추출"""
    terms = []
    zip_paths = [
        AIHUB_DIR / "Training/02.라벨링데이터/TL_02 용어사전.zip",
        AIHUB_DIR / "Validation/02.라벨링데이터/VL_02 용어사전.zip",
    ]
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                obj = json.loads(z.read(name).decode("utf-8"))
                for entry in obj["data"]:
                    if entry.get("domain") == "의학":
                        terms.append(
                            {
                                "ko": entry["source_word"].strip(),
                                "en": entry["target_word"].strip(),
                                "code": "",
                                "umls": "",
                                "snomed": "",
                                "domain": "의학용어(일반)",
                                "source": f"AIHub_{zip_path.name}",
                            }
                        )
    return terms


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    kcd_terms = load_kcd_terms()
    print(f"KCD 상병마스터 질병명 용어: {len(kcd_terms)}건")

    kostom_terms = load_kostom_terms()
    print(f"KOSTOM 진단 관련 용어(KCD 코드 보유): {len(kostom_terms)}건")

    aihub_terms = load_aihub_medical_terms()
    print(f"AI Hub 의학 도메인 용어: {len(aihub_terms)}건")

    # 한글명 기준 중복 제거, 먼저 나온 항목 우선순위로 채택
    # 우선순위: KCD상병마스터(공식 질병분류명) > KOSTOM(UMLS/SNOMED 포함, 커버리지 넓음) > AIHub(일반 의학용어, 검수 안 된 자료)
    all_terms = kcd_terms + kostom_terms + aihub_terms
    seen = set()
    deduped = []
    for t in all_terms:
        key = t["ko"]
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    print(f"병합 전 총합: {len(all_terms)}건 -> 한글명 중복 제거 후: {len(deduped)}건")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"저장 완료: {OUT_CSV}")


if __name__ == "__main__":
    main()
