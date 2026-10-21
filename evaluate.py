# -*- coding: utf-8 -*-
"""
OCR-ის ხარისხის გაზომვა ხელით აკრეფილ ტექსტთან (ground truth) შედარებით.

ground_truth საქაღალდეში ფაილის სახელი: <PDF-ის სახელი>_p<გვერდი>.txt, მაგ.
    Guriis_Erobis_Moambe_1920_N1_p001.txt
ფაილის თითოეული სტრიქონი გვერდის ერთი მონაკვეთია (მაგ. აბზაცი) - გვერდზე ნებისმიერ ადგილას
შეიძლება იყოს; სკრიპტი OCR-ის ტექსტში საუკეთესო შესაბამის ადგილს თვითონ პოულობს.
მთელი გვერდის აკრეფა საჭირო არ არის - 2-3 აბზაცი თითო გვერდზე საკმარისია.

    python evaluate.py extracted_texts               # ground_truth/ საქაღალდესთან შედარება
    python evaluate.py extracted_texts old_outputs   # ორი ვერსიის შედარება

CER = სიმბოლოების შეცდომა (%), WER = სიტყვების შეცდომა (%). რაც ნაკლებია, მით უკეთესი.
"""

import os
import re
import sys

import numpy as np

GT_FOLDER = "ground_truth"


def norm(t):
    t = t.replace("­", "")
    return re.sub(r"\s+", " ", t).strip()


def semiglobal(gt, ocr):
    """რედაქტირების მანძილი gt-სა და ocr-ის საუკეთესოდ შესაბამის მონაკვეთს შორის."""
    m = len(ocr)
    o = np.array([ord(c) for c in ocr], dtype=np.int64)
    prev = np.zeros(m + 1, dtype=np.int64)
    for i, ch in enumerate(gt, 1):
        best = np.minimum(prev[:-1] + (o != ord(ch)), prev[1:] + 1)
        cur = np.empty(m + 1, dtype=np.int64)
        cur[0] = i
        # ჩასმა: cur[j] = min(best[j-1], cur[j-1] + 1)  ->  min(best[k] + (j-1-k)), k < j
        idx = np.arange(m)
        cur[1:] = np.minimum.accumulate(best - idx) + idx
        cur[1:] = np.minimum(cur[1:], i + idx + 1)
        prev = cur
    return int(prev.min())


def segments(gt):
    return [norm(s) for s in gt.split("\n") if s.strip()]


def scores(gt, ocr):
    ocr = norm(ocr)
    segs = segments(gt)
    n = sum(len(s) for s in segs)
    c = sum(semiglobal(s, ocr) for s in segs)
    words = ocr.split()
    ws = [s.split() for s in segs]
    ids = {w: i for i, w in enumerate(set(words) | {w for s in ws for w in s})}
    enc = lambda seq: "".join(chr(0x4E00 + ids[w]) for w in seq)
    nw = sum(len(s) for s in ws)
    e = sum(semiglobal(enc(s), enc(words)) for s in ws)
    return c, n, e, nw


def page_text(path, page):
    if not os.path.exists(path):
        return None
    out, on = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^===== გვერდი (\d+) =====", line)
            if m:
                on = int(m.group(1)) == page
                continue
            if on:
                out.append(line)
    return "".join(out)


def main():
    folders = sys.argv[1:] or ["extracted_texts"]
    gts = sorted(f for f in os.listdir(GT_FOLDER) if f.endswith(".txt"))
    if not gts:
        sys.exit(f"{GT_FOLDER} საქაღალდეში ფაილები არ არის.")
    tot = {f: [0, 0, 0, 0] for f in folders}
    print(f"{'გვერდი':45s}" + "".join(f"{os.path.basename(os.path.normpath(f)):>28s}" for f in folders))
    for g in gts:
        m = re.match(r"(.+)_p(\d+)\.txt$", g)
        if not m:
            continue
        stem, page = m.group(1), int(m.group(2))
        gt = open(os.path.join(GT_FOLDER, g), encoding="utf-8").read()
        row = f"{g[:-4]:45s}"
        for f in folders:
            t = page_text(os.path.join(f, stem + ".txt"), page)
            if t is None:
                row += f"{'-':>28s}"
                continue
            c, n, e, nw = scores(gt, t)
            for k, v in enumerate((c, n, e, nw)):
                tot[f][k] += v
            row += f"{'CER %5.2f%%  WER %5.2f%%' % (100 * c / n, 100 * e / nw):>28s}"
        print(row)
    row = f"{'სულ':45s}"
    for f in folders:
        c, n, e, nw = tot[f]
        row += f"{'CER %5.2f%%  WER %5.2f%%' % (100 * c / max(1, n), 100 * e / max(1, nw)):>28s}"
    print(row)


if __name__ == "__main__":
    main()
