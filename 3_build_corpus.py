import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT_DIR = Path("data/corpus")


def esearch(term, retmax=200):
    url = f"{EUTILS}/esearch.fcgi?db=pubmed&term={urllib.parse.quote(term)}&retmax={retmax}&retmode=json"
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    return data["esearchresult"]["idlist"]


def efetch_abstracts(pmids):
    ids = ",".join(pmids)
    url = f"{EUTILS}/efetch.fcgi?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def parse_abstracts(xml_bytes):
    root = ET.fromstring(xml_bytes)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID", default="")
        title = art.findtext(".//ArticleTitle", default="")
        abstract_parts = [
            (node.text or "") for node in art.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(abstract_parts).strip()
        mesh_terms = [
            m.text for m in art.findall(".//MeshHeading/DescriptorName") if m.text
        ]
        if abstract:
            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "mesh_terms": mesh_terms,
                }
            )
    return articles


def build_corpus(query, out_name, retmax=200):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PubMed 검색: {query}")
    pmids = esearch(query, retmax=retmax)
    print(f"검색된 PMID: {len(pmids)}건")

    time.sleep(0.4)  # NCBI 요청 간격 권장(무인증 시 초당 3회 이하)
    xml_bytes = efetch_abstracts(pmids)
    articles = parse_abstracts(xml_bytes)
    print(f"초록 있는 문서: {len(articles)}건 (제목만 있고 초록 없는 문서는 제외)")

    out_path = OUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as fp:
        for a in articles:
            fp.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"저장 완료: {out_path}")
    return articles


if __name__ == "__main__":
    build_corpus(
        query="diabetes mellitus[MeSH Major Topic] AND review[Publication Type]",
        out_name="pubmed_diabetes.jsonl",
        retmax=200,
    )
