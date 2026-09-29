import json
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

BASE_URL = (
    "https://health.kdca.go.kr/healthinfo/biz/health/gnrlzHealthInfo/"
    "gnrlzHealthInfo/gnrlzHealthInfoView.do?cntnts_sn={sn}"
)
OUT_PATH = Path("data/corpus/kdca_diabetes_kr.jsonl")

# 국가건강정보포털(질병관리청) 당뇨병 관련 페이지
# 라이선스: 공공누리 제4유형(출처표시 + 비상업적 이용 + 변경금지) -> 원문 그대로 저장, 요약/편집 금지
PAGES = [
    (5305, "당뇨병"),
    (2347, "당뇨병 합병증(급성 합병증)"),
    (2351, "당뇨병 합병증(만성 합병증)"),
    (3388, "당뇨환자의 식이요법"),
    (3390, "당뇨환자의 운동요법"),
    (6471, "당뇨환자의 약물요법"),
]

LICENSE_NOTE = "공공누리 제4유형(출처표시-비상업적이용-변경금지). 출처: 질병관리청 국가건강정보포털"


def fetch_content(sn):
    url = BASE_URL.format(sn=sn)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")

    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", class_="hd-contents")
    text = div.get_text(separator="\n", strip=True) if div else ""
    return url, text


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    for sn, title in PAGES:
        print(f"수집 중: [{sn}] {title}")
        url, text = fetch_content(sn)
        if not text:
            print(f"  경고: 본문을 찾지 못함 (cntnts_sn={sn})")
            continue
        docs.append(
            {
                "cntnts_sn": sn,
                "title": title,
                "url": url,
                "content": text,
                "license": LICENSE_NOTE,
            }
        )
        time.sleep(0.5)  # 서버 부담 최소화

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        for d in docs:
            fp.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n저장 완료: {OUT_PATH} ({len(docs)}건)")


if __name__ == "__main__":
    main()
