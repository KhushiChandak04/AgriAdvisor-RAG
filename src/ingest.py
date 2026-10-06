"""Build a FRESH vectorstore from the crop PDFs in data/.

Run:  python -m src.ingest
Any existing vectorstore (e.g. the old HR-policy one) is deleted first.
Only PDFs whose filename contains rice / wheat / maize / cotton / soybean / sugarcane are used.
"""
import json
import shutil
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .pdf_processing import chunk_page, extract_pages


def ingest():
    pdfs = sorted({p for p in config.DATA_DIR.iterdir() if p.suffix.lower() == ".pdf"})
    selected, rejected = [], []
    for p in pdfs:
        crop = config.detect_crop(p.stem)
        (selected if crop else rejected).append((p, crop))

    for p, _ in rejected:
        print(f"[SKIP] {p.name}: filename has no crop name "
              f"({', '.join(config.CROPS)}). Rename it, e.g. 'wheat_{p.stem}.pdf'.")
    if not selected:
        print("\nERROR: no usable PDFs in data/. Add agriculture PDFs named like "
              "rice_guide.pdf, wheat_pop.pdf, ... (see README step 4).")
        sys.exit(1)

    # Fresh start: wipe any old index (HR-policy leftovers etc.)
    if config.VECTORSTORE_DIR.exists():
        shutil.rmtree(config.VECTORSTORE_DIR)
    config.VECTORSTORE_DIR.mkdir(parents=True)
    print("Old vectorstore deleted. Building a fresh one.\n")

    chunks = []
    crop_chunks = {crop: 0 for crop in config.CROPS}
    
    for pdf, crop in selected:
        print(f"[{crop}] {pdf.name}")
        pages = extract_pages(pdf)
        n_before = len(chunks)
        for pg in pages:
            for item in chunk_page(pg["text"]):
                chunks.append({"text": item["text"], "source": pdf.name, "page": pg["page"],
                               "section": item["section"], "crop": crop, "ocr": pg["ocr"]})
                crop_chunks[crop] += 1
        print(f"    {len(pages)} pages -> {len(chunks) - n_before} chunks (total: {crop_chunks[crop]})")

    if not chunks:
        print("ERROR: no text could be extracted (scanned PDFs need Tesseract OCR).")
        sys.exit(1)

    print(f"\n--- Per-crop chunk counts ---")
    for crop in config.CROPS:
        count = crop_chunks[crop]
        status = "✓" if count >= 150 else "⚠️ LOW"
        print(f"{crop:12} {count:4} chunks {status}")
    
    low_crops = [c for c in config.CROPS if crop_chunks[c] > 0 and crop_chunks[c] < 150]
    if low_crops:
        print(f"\n⚠️  WARNING: {', '.join(low_crops)} have <150 chunks. Consider adding more PDFs for better coverage.")

    print(f"\nEmbedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    emb = model.encode([c["text"] for c in chunks], batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity (vectors are normalized)
    index.add(emb)
    faiss.write_index(index, str(config.VECTORSTORE_DIR / "index.faiss"))
    (config.VECTORSTORE_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (config.VECTORSTORE_DIR / "meta.json").write_text(json.dumps({
        "domain": "agriculture",
        "model": config.EMBEDDING_MODEL,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_chunks": len(chunks),
        "files": sorted({c["source"] for c in chunks}),
        "crops": sorted({c["crop"] for c in chunks}),
        "crop_chunks": crop_chunks,
    }, indent=2), encoding="utf-8")

    print(f"\nDone. {len(chunks)} chunks from {len({c['source'] for c in chunks})} PDFs "
          f"saved to vectorstore/")
    present = {x["crop"] for x in chunks}
    missing = [c for c in config.CROPS if c not in present]
    if missing:
        print(f"Note: no PDFs yet for: {', '.join(missing)}")


if __name__ == "__main__":
    ingest()
