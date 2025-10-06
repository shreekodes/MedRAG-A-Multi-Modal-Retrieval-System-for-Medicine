import requests, json, os
from bs4 import BeautifulSoup
from time import sleep

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
API_KEY = ""  # optional

def fetch_pubmed(term, retmax=20):
    params = {"db":"pubmed","term":term,"retmode":"json","retmax":retmax}
    if API_KEY: params["api_key"] = API_KEY
    r = requests.get(ESEARCH, params=params); r.raise_for_status()
    ids = r.json().get("esearchresult",{}).get("idlist", [])
    if not ids: return []
    params2 = {"db":"pubmed","id":",".join(ids),"retmode":"xml"}
    if API_KEY: params2["api_key"] = API_KEY
    r2 = requests.get(EFETCH, params=params2); r2.raise_for_status()
    soup = BeautifulSoup(r2.text, "xml")
    out=[]
    for art in soup.find_all("PubmedArticle"):
        pmid = art.find("PMID").text if art.find("PMID") else ""
        title = art.ArticleTitle.text if art.ArticleTitle else ""
        abstract = ""
        abs_tag = art.find("Abstract")
        if abs_tag:
            abstract = " ".join([p.text for p in abs_tag.find_all("AbstractText")])
        out.append({"pmid":pmid, "title":title, "abstract":abstract})
    return out

if __name__ == "__main__":
    terms = ["glioblastoma MRI", "meningioma MRI", "Alzheimer MRI", "stroke MRI"]
    all_docs=[]
    for t in terms:
        docs = fetch_pubmed(t, retmax=5)
        for d in docs:
            d["topic"] = t.split()[0].lower()
        all_docs += docs
        sleep(1)  # polite
    os.makedirs("data/text_samples", exist_ok=True)
    with open("data/text_samples/abstracts_neuro.json","w",encoding="utf-8") as f:
        json.dump(all_docs,f,indent=2,ensure_ascii=False)
    print("Saved", len(all_docs), "abstracts")
