"""Download crop PDFs straight into data/.

Usage (run from the project folder, venv active):
  python scripts\download_pdfs.py                      -> reads scripts\pdf_links.txt
  python scripts\download_pdfs.py rice https://site/rice.pdf   -> one link directly

pdf_links.txt format (one per line):  crop  URL
Lines starting with # are ignored.
"""
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LINKS_FILE = Path(__file__).parent / "pdf_links.txt"
CROPS = {"rice", "wheat", "maize", "cotton", "soybean", "sugarcane"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def download(crop, url):
    r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
    r.raise_for_status()
    if r.content[:5] != b"%PDF-":
        return False
    name = unquote(Path(urlparse(r.url).path).name) or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    out = DATA / f"{crop}_{name}"
    out.write_bytes(r.content)
    print(f"   OK  {out.name}  ({len(r.content) // 1024} KB)")
    return True


def handle(crop, url):
    crop = crop.lower()
    if crop not in CROPS:
        print(f"   SKIP unknown crop '{crop}' (use: {', '.join(sorted(CROPS))})")
        return 0
    print(f"[{crop}] {url}")
    try:
        if download(crop, url):
            return 1
        # Not a PDF: it is a web page. Look for PDF links on it.
        page = requests.get(url, headers=HEADERS, timeout=60)
        soup = BeautifulSoup(page.text, "html.parser")
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)
                 if a["href"].lower().split("?")[0].endswith(".pdf")][:3]
        if not links:
            print("   FAIL not a PDF and no PDF links found. Use a direct .pdf link.")
            return 0
        return sum(1 for l in links if download(crop, l))
    except Exception as e:
        print(f"   FAIL {e}")
        return 0


def main():
    DATA.mkdir(exist_ok=True)
    jobs = []
    if len(sys.argv) == 3:
        jobs.append((sys.argv[1], sys.argv[2]))
    elif LINKS_FILE.exists():
        for line in LINKS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    jobs.append((parts[0], parts[1].strip()))
    else:
        print("Usage: python scripts\\download_pdfs.py <crop> <url>   or fill scripts\\pdf_links.txt")
        return

    ok = sum(handle(c, u) for c, u in jobs)
    print(f"\nDownloaded {ok} PDF(s) into data\\")
    print("Next: run_ingest.bat")


if __name__ == "__main__":
    main()