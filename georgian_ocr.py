# -*- coding: utf-8 -*-
"""
ძველი ქართული გაზეთების / ჟურნალების PDF-ების OCR (Tesseract, ქართული) - ვერსია 6.

წინა ვერსიებისგან განსხვავება:
  * გვერდს ყოფს სვეტებად (XY-cut): ჭრის მხოლოდ ნამდვილ გამყოფ ხაზებზე და სვეტებს შორის
    თეთრ ზოლებზე. სიტყვებს შორის ფართო დაშორება (გასწორებული ტექსტი) სვეტს აღარ ჭრის.
  * სვეტს Tesseract მთლიანად კითხულობს - სათაური და მისი ტექსტი ერთად რჩება.
  * სკანის საკუთარ რეზოლუციაზე მუშაობს (ზედმეტი გადიდება ამ სკანებზე ცნობას აუარესებს).
  * სტრიქონის ბოლოს გაწყვეტილ სიტყვებს აერთებს სიტყვების ლექსიკონით, მაშინაც კი,
    როცა დეფისი ვერ ამოიცნო (ან "."-დ წაიკითხა). სვეტიდან სვეტზე და გვერდიდან გვერდზეც.
  * თითოეულ სტრიქონს სამი ვარიანტიდან (ნაცრისფერი / შავ-თეთრი / გადიდებული) ირჩევს -
    სანდოობით და იმით, რამდენი სიტყვაა ლექსიკონში.
  * სტრიქონის ბოლოს "მოგონილ" წერტილებს და ლაქებისგან გაჩენილ ბრჭყალებს სურათზე ამოწმებს.
  * დაბალი სანდოობის ტექსტს აღარ აგდებს - მხოლოდ ნამდვილ "ნაგავს" (ორნამენტი, შტამპი,
    გაზეთის სახელი დეკორატიული შრიფტით).
  * აბზაცებს და სათაურებს სტრიქონების მდებარეობით ცნობს.

გაშვება:
    python georgian_ocr.py                      # pdf_files საქაღალდის ყველა PDF
    python georgian_ocr.py ფაილი.pdf --pages 1-2 --debug
"""

import argparse
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================ პარამეტრები ============================

# Tesseract-ის გზა (Linux/Mac-ზე უბრალოდ "tesseract")
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# tessdata_best მოდელების საქაღალდე (kat.traineddata https://github.com/tesseract-ocr/tessdata_best-დან).
# ცარიელი = Tesseract-ის ჩვეულებრივი tessdata
TESSDATA_DIR = ""

INPUT_PDF_FOLDER = "pdf_files"
OUTPUT_TXT_FOLDER = "extracted_texts"

# ენა. "kat+rus" - თუ ბევრი რუსული ჩანართია
LANG = "kat"

# ქართული სიტყვების სია (თითო სიტყვა სტრიქონზე) - დეფისით გაწყვეტილი სიტყვების აღსადგენად.
# სკრიპტთან ერთად მოყვება (ამოღებულია Tesseract-ის kat მოდელიდან). დამატებით სიტყვებს
# OUTPUT_TXT_FOLDER-ში არსებული შედეგებიდანაც სწავლობს.
WORDLIST = "georgian_words.txt"

# "best" - თითოეულ სტრიქონს სამ ვარიანტად კითხულობს და უკეთესს ირჩევს (≈3-ჯერ ნელი)
# "fast" - ერთხელ
QUALITY = "best"

# სტრიქონის ბიჯი (ერთი სტრიქონის ფუძიდან მეორემდე) პიქსელებში. თუ სკანზე ბიჯი ამ
# საზღვრებშია, სურათი არ იცვლება; თუ არა - მოერგება TARGET_LINE_PITCH-ს.
MIN_LINE_PITCH = 18
MAX_LINE_PITCH = 48
TARGET_LINE_PITCH = 28

# გვერდის ზედა სტრიქონი (გაზეთის სახელი, გვერდის ნომერი, თარიღი) - ტექსტში არ გავიდეს
DROP_PAGE_HEADERS = True
# სათაურებს "## "-ით აღნიშნავს
MARK_HEADINGS = False
# თითოეული სტრიქონი ცალკე (აბზაცებად გაერთიანების და დეფისების აღდგენის გარეშე)
KEEP_LINES = False

DEBUG_IMAGES = False    # დაჭრის სქემა: extracted_texts/_debug/
SKIP_EXISTING = True    # უკვე დამუშავებულ PDF-ებს გამოტოვებს
WORKERS = max(1, (os.cpu_count() or 2) - 1)   # რამდენი გვერდი ერთდროულად

# =====================================================================

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # Tesseract-ის ნაკადები ერთმანეთს არ უნდა შეეჯახოს

import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract

GEO = "\u10D0-\u10FF"
HYPHENS = "-\u2010\u2011\u2012\u00AC~="          # მოკლე დეფისები (და ის, რადაც Tesseract კითხულობს)
DASHES = "\u2013\u2014\u2015"                     # – — ―
OPEN_Q, CLOSE_Q = "„«", "”»"
MAX_PIXELS = 40_000_000

# ---------------------------- გვერდის ჩატვირთვა ----------------------------


def load_page(page):
    """გვერდი სკანის საკუთარი რეზოლუციით (ან 300 DPI, თუ სკანი არ არის)."""
    zoom = 300 / 72
    best = 0
    for im in page.get_images(full=True):
        try:
            rects = page.get_image_rects(im[0])
        except Exception:
            rects = []
        for r in rects:
            if r.width > 0 and r.width * r.height > best:
                best = r.width * r.height
                zoom = im[2] / r.width
    zoom = min(zoom, (MAX_PIXELS / max(1.0, page.rect.width * page.rect.height)) ** 0.5)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()


# ---------------------------- სურათის მომზადება ----------------------------


def normalize(gray):
    """ფონი თეთრდება (გაყვითლება, ჩრდილი, ლაქები), შემდეგ კონტრასტი იჭიმება."""
    h, w = gray.shape
    f = 8
    small = cv2.resize(gray, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255.0, 0, 255)
    lo, hi = np.percentile(norm[::2, ::2], [1, 60])     # მუქი მელანი -> შავი, ქაღალდი -> თეთრი
    norm = (norm - lo) / max(1.0, hi - lo) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def sauvola(gray, win, k=0.2, R=128.0):
    """აბრუნებს ink ნიღაბს (255 = მელანი)."""
    win = max(15, int(win)) | 1
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, cv2.CV_32F, (win, win))
    s = cv2.boxFilter(g * g, cv2.CV_32F, (win, win))
    s = np.sqrt(np.maximum(s - m * m, 0))
    t = m * (1 + k * (s / R - 1))
    return ((g < t) & (g < 200)).astype(np.uint8) * 255


def remove_small(mask, min_area):
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def line_pitch(ink):
    """სტრიქონის ბიჯი: ავტოკორელაცია ვიწრო ვერტიკალურ ზოლებში (სვეტების შიგნით)."""
    H, W = ink.shape
    found = []
    sw = max(40, W // 10)
    for x0 in range(0, W - sw + 1, max(1, sw // 2)):
        v = (ink[:, x0:x0 + sw] > 0).sum(1).astype(np.float32)
        if v.sum() < 50 * sw / 10:
            continue
        v = cv2.GaussianBlur(v.reshape(-1, 1), (1, 0), 1.5).ravel()
        v -= v.mean()
        ac = np.correlate(v, v, "full")[len(v) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = 8, min(len(ac) - 1, 160)
        if hi <= lo + 2:
            continue
        seg = ac[lo:hi]
        # პირველი მკაფიო მწვერვალი (არა უბრალოდ მაქსიმუმი - ორმაგი ბიჯიც მაღალია)
        peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.25]
        if peaks:
            k = lo + peaks[0]
            found.append((float(ac[k]), k))
    if not found:
        return 0.0
    found.sort(reverse=True)
    return float(np.median([k for _, k in found[:max(3, len(found) // 2)]]))


def deskew_angle(ink, max_angle=3.0):
    """გვერდის დახრა სტრიქონების პროექციით (უხეშად, შემდეგ ზუსტად)."""
    h, w = ink.shape
    s = min(1.0, 1200.0 / w)
    small = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape

    def score(a):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), a, 1.0)
        r = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(r.sum(axis=1, dtype=np.float64)))

    best = max(np.arange(-max_angle, max_angle + 0.01, 0.5), key=score)
    best = max(np.arange(best - 0.4, best + 0.41, 0.1), key=score)
    return float(best) if abs(best) >= 0.1 else 0.0


def rotate(img, angle, border):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def prepare(gray):
    """აბრუნებს (norm, ink, P): გასწორებულ ნაცრისფერ სურათს, მელნის ნიღაბს, სტრიქონის ბიჯს."""
    norm = normalize(gray)
    ink = sauvola(norm, 41)
    P = line_pitch(ink) or 30.0
    if not MIN_LINE_PITCH <= P <= MAX_LINE_PITCH:
        scale = TARGET_LINE_PITCH / P
        scale = min(scale, (MAX_PIXELS / (gray.shape[0] * gray.shape[1])) ** 0.5)
        scale = max(0.3, min(3.0, scale))
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        norm = normalize(cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp))
        P *= scale
    ink = sauvola(norm, 1.6 * P)
    a = deskew_angle(ink)
    if a:
        norm = rotate(norm, a, 255)
        ink = sauvola(norm, 1.6 * P)
    return norm, ink, P


# ---------------------------- გამყოფი ხაზები ----------------------------


def _keep_lines(mask, axis, min_len, max_thick):
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    length = st[:, cv2.CC_STAT_HEIGHT] if axis == 0 else st[:, cv2.CC_STAT_WIDTH]
    keep = (length >= min_len) & (st[:, cv2.CC_STAT_AREA] / np.maximum(length, 1) <= max_thick)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def find_rules(ink, P):
    """მხოლოდ გრძელი და წვრილი ხაზები - დიდი ასოების შტრიხები ამ პირობას ვერ აკმაყოფილებს."""
    H, W = ink.shape
    t = max(3, int(0.12 * P)) | 1
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (t, 1)))
    v = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(3 * P))))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(1.5 * P))))
    v = _keep_lines(v, 0, max(6 * P, 0.06 * H), 0.45 * P)

    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, t)))
    h = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * P), 1)))
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(1.0 * P), 1)))
    h = _keep_lines(h, 1, max(4 * P, 0.05 * W), 0.45 * P)
    return v, h


# ---------------------------- სვეტები და ბლოკები (XY-cut) ----------------------------


def _runs(flags):
    """True-ების უწყვეტი მონაკვეთები: [(დასაწყისი, დასასრული)]."""
    d = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


class Layout:
    def __init__(self, mask, vr, hr, P):
        self.mask, self.hr, self.P = mask, hr, P
        # ოდნავ დახრილი/მრუდე ხაზი ერთ პიქსელურ სვეტში მთლიანად არ ჯდება - ვაგანიერებთ
        k = max(3, int(0.4 * P)) | 1
        self.vr = cv2.dilate(vr, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        self.leaves = []

    def tight(self, x0, y0, x1, y1):
        sub = self.mask[y0:y1, x0:x1]
        rows, cols = np.where(sub.any(axis=1))[0], np.where(sub.any(axis=0))[0]
        if not len(rows):
            return None
        return x0 + cols[0], y0 + rows[0], x0 + cols[-1] + 1, y0 + rows[-1] + 1

    def run(self):
        H, W = self.mask.shape
        self.rec(0, 0, W, H, 0)
        return self.leaves

    def rec(self, x0, y0, x1, y1, depth):
        r = self.tight(x0, y0, x1, y1)
        if r is None:
            return
        x0, y0, x1, y1 = r
        if depth > 40:
            self.leaves.append(r)
            return
        for finder, axis in ((self.rule_vcuts, 0), (self.rule_hcuts, 1), (self.gap_vcuts, 0), (self.partial_hcuts, 1)):
            cuts = finder(x0, y0, x1, y1)
            if cuts:
                edges = [x0 if axis == 0 else y0] + sorted(cuts) + [x1 if axis == 0 else y1]
                for a, b in zip(edges[:-1], edges[1:]):
                    if axis == 0:
                        self.rec(a, y0, b, y1, depth + 1)
                    else:
                        self.rec(x0, a, x1, b, depth + 1)
                return
        self.leaves.append(r)

    # --- 1) ხაზი, რომელიც ბლოკს თითქმის მთელ სიმაღლეზე/სიგანეზე კვეთს
    def rule_vcuts(self, x0, y0, x1, y1):
        h = y1 - y0
        if h < 3 * self.P:
            return []
        cov = (self.vr[y0:y1, x0:x1] > 0).sum(axis=0)
        cuts = []
        for a, b in _runs(cov >= 0.8 * h):
            c = x0 + (a + b) // 2
            if c - x0 > 2 * self.P and x1 - c > 2 * self.P:
                cuts.append(c)
        return cuts

    def rule_hcuts(self, x0, y0, x1, y1):
        w = x1 - x0
        cov = (self.hr[y0:y1, x0:x1] > 0).sum(axis=1)
        cuts = []
        for a, b in _runs(cov >= 0.8 * w):
            c = y0 + (a + b) // 2
            if c - y0 > 0.5 * self.P and y1 - c > 0.5 * self.P:
                cuts.append(c)
        return cuts

    # --- 2) სვეტებს შორის თეთრი ზოლი მთელ სიმაღლეზე (ან ნაწილობრივ ჩაკეტილი ხაზი)
    def _gutters(self, x0, y0, x1, y1, tol_frac, min_w):
        P = self.P
        sub = self.mask[y0:y1, x0:x1] > 0
        h = y1 - y0
        ink = sub.sum(axis=0)
        vcov = (self.vr[y0:y1, x0:x1] > 0).sum(axis=0) >= 0.5 * h
        empty = (ink <= max(1, tol_frac * h)) | vcov
        out = []
        for a, b in _runs(empty):
            if a == 0 or b == len(empty):
                continue
            if (b - a) < min_w and not vcov[a:b].any():
                continue
            left, right = sub[:, :a].sum(), sub[:, b:].sum()
            if a < 3 * P or (len(empty) - b) < 3 * P or left < 2 * P * P or right < 2 * P * P:
                continue
            out.append(x0 + (a + b) // 2)
        return out

    def gap_vcuts(self, x0, y0, x1, y1):
        h = y1 - y0
        P = self.P
        if h < 2.5 * P:
            min_w = 3.0 * P          # ერთი-ორი სტრიქონი: სიტყვებს შორის არ გაჭრას
        elif h < 8 * P:
            min_w = 1.6 * P
        else:
            min_w = 1.0 * P
        return self._gutters(x0, y0, x1, y1, 0.004, min_w)

    # --- 3) სვეტების გამყოფს ზოგან რაღაც კვეთს (რამდენიმე სვეტზე გადაჭიმული სათაური,
    #        გაზეთის სახელი...) - ვჭრით ჰორიზონტალურად სწორედ იმ ადგილის ზემოთ/ქვემოთ
    def partial_hcuts(self, x0, y0, x1, y1):
        P = self.P
        h, w = y1 - y0, x1 - x0
        if h < 6 * P or w < 8 * P:
            return []
        sub = self.mask[y0:y1, x0:x1] > 0
        vsub = self.vr[y0:y1, x0:x1] > 0
        band = max(4, int(P))
        nb = h // band
        if nb < 4:
            return []
        gw = max(3, int(1.0 * P))
        # თითოეულ ზოლში: რომელი სვეტებია ცარიელი
        col_ink = sub[:nb * band].reshape(nb, band, w).sum(1)        # nb x w
        rule = vsub[:nb * band].reshape(nb, band, w).any(1)
        k = np.ones(gw)
        win = np.stack([np.convolve(r, k, "same") for r in col_ink])   # ფანჯრის ჯამი
        free = (win <= 1) | rule
        frac = free.mean(0)
        margin = int(3 * P)
        if w <= 2 * margin:
            return []
        xs = np.arange(margin, w - margin)
        x = int(xs[np.argmax(frac[margin:w - margin])])
        if frac[x] < 0.55 or frac[x] >= 0.999:
            return []
        # ორივე მხარეს ტექსტი უნდა იყოს იმ ზოლებში, სადაც გამყოფი თავისუფალია
        fb = free[:, x]
        if sub[:nb * band][np.repeat(fb, band)][:, :x].sum() < 2 * P * P or \
           sub[:nb * band][np.repeat(fb, band)][:, x:].sum() < 2 * P * P:
            return []
        blocked = _runs(~fb)
        rowink = sub.sum(axis=1)
        empty_row = rowink <= max(1, 0.003 * w)
        cuts = []
        for a, b in blocked:
            ya, yb = a * band, min(h, b * band)
            # უახლოესი ცარიელი სტრიქონი ზემოთ და ქვემოთ
            up = np.where(empty_row[:ya])[0]
            dn = np.where(empty_row[yb:])[0]
            if a > 0 and len(up):
                cuts.append(y0 + int(up[-1]))
            if b < nb and len(dn):
                cuts.append(y0 + yb + int(dn[0]))
        cuts = sorted(set(c for c in cuts if y0 + 0.3 * P < c < y1 - 0.3 * P))
        return cuts


def segment(ink, P):
    vr, hr = find_rules(ink, P)
    rules = cv2.dilate(vr | hr, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(ink, cv2.bitwise_not(rules))
    mask = remove_small(mask, max(4, int(0.012 * P * P)))
    # დიდი, თითქმის შავი ლაქები (სურათი, მელნის ლაქა) - ბლოკებს არ უნდა აერთებდეს
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bw, bh, ar = st[:, 2], st[:, 3], st[:, 4]
    blot = (ar > 40 * P * P) & (ar > 0.6 * bw * bh)
    blot[0] = False
    if blot.any():
        mask[blot[labels]] = 0
    leaves = Layout(mask, vr, hr, P).run()
    return leaves, vr, hr, mask


# ---------------------------- OCR ----------------------------


_LEX = None


def lexicon():
    global _LEX
    if _LEX is None:
        _LEX = load_wordlist()
    return _LEX


def load_wordlist():
    words = set()
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (WORDLIST, os.path.join(here, WORDLIST)):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                words.update(w.strip() for w in f if w.strip())
            break
    return words


def tess_config(psm):
    return f"--oem 1 --psm {psm}"


def use_tessdata():
    # --tessdata-dir-ის ნაცვლად გარემოს ცვლადი: Windows-ზე გზაში ჰარი/ბრჭყალები არ ირევა
    if TESSDATA_DIR:
        os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


def ocr_lines(img, psm, ox, oy, scale):
    """Tesseract -> სტრიქონები [{text, words:[(text,conf,x0,y0,x1,y1)], box}] გვერდის კოორდინატებში."""
    pad = 20
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(img, lang=LANG, config=tess_config(psm), output_type=pytesseract.Output.DICT)
    lines, cur, key = [], [], None
    for i, txt in enumerate(d["text"]):
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key:
            if cur:
                lines.append(cur)
            cur, key = [], k
        if txt.strip():
            x0 = (d["left"][i] - pad) / scale + ox
            y0 = (d["top"][i] - pad) / scale + oy
            cur.append((txt.strip(), float(d["conf"][i]), x0, y0,
                        x0 + d["width"][i] / scale, y0 + d["height"][i] / scale))
    if cur:
        lines.append(cur)
    out = []
    for ws in lines:
        out.append(dict(words=ws, box=(min(w[2] for w in ws), min(w[3] for w in ws),
                                       max(w[4] for w in ws), max(w[5] for w in ws))))
    return out


def word_core(w):
    m = re.search(rf"[{GEO}]+", w)
    return m.group(0) if m else ""


def line_score(line):
    """ორი ვარიანტიდან ასარჩევად: სანდოობა + რამდენი სიტყვაა ლექსიკონში."""
    lex = lexicon()
    ws = line["words"]
    n = sum(len(w[0]) for w in ws)
    if not n:
        return -1.0
    conf = sum(max(0.0, w[1]) * len(w[0]) for w in ws) / n
    cores = [word_core(w[0]) for w in ws]
    cores = [c for c in cores if len(c) >= 2]
    known = (sum(c in lex for c in cores) / len(cores)) if cores and lex else 0.0
    return conf + 40.0 * known


def merge_variants(a, b):
    """a-ს თითოეულ სტრიქონს b-ს შესაბამისი სტრიქონით ცვლის, თუ ის უკეთესია."""
    used = set()
    out = []
    for la in a:
        ay0, ay1 = la["box"][1], la["box"][3]
        best, bo = None, 0.0
        for j, lb in enumerate(b):
            if j in used:
                continue
            ov = min(ay1, lb["box"][3]) - max(ay0, lb["box"][1])
            ovx = min(la["box"][2], lb["box"][2]) - max(la["box"][0], lb["box"][0])
            r = ov / max(1.0, min(ay1 - ay0, lb["box"][3] - lb["box"][1]))
            if r > bo and ovx > 0:
                best, bo = j, r
        if best is None or bo < 0.6:
            out.append(la)
            continue
        used.add(best)
        out.append(b[best] if line_score(b[best]) > line_score(la) + 2 else la)
    return out


def ocr_block(norm, rules, box, P):
    x0, y0, x1, y1 = box
    H, W = norm.shape
    m = 2
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    crop = norm[y0:y1, x0:x1].copy()
    crop[rules[y0:y1, x0:x1] > 0] = 255
    psm = 7 if (y1 - y0) < 1.8 * P else 6
    lines = ocr_lines(crop, psm, x0, y0, 1.0)
    if QUALITY == "best":
        # იგივე ბლოკი შავ-თეთრად და 1.5-ჯერ გადიდებული; თითოეული სტრიქონისთვის უკეთესი რჩება
        binv = 255 - remove_small(sauvola(crop, 1.6 * P), 3)
        lines = merge_variants(lines, ocr_lines(binv, psm, x0, y0, 1.0))
        big = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        lines = merge_variants(lines, ocr_lines(big, psm, x0, y0, 1.5))
    return lines


# ---------------------------- გვერდი ----------------------------


def process_page(args):
    pdf_path, pno, debug_dir = args
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    doc = fitz.open(pdf_path)
    gray = load_page(doc[pno])
    doc.close()
    norm, ink, P = prepare(gray)
    leaves, vr, hr, mask = segment(ink, P)
    rules = cv2.dilate(vr | hr, np.ones((3, 3), np.uint8))
    H, W = norm.shape
    blocks = []
    for box in leaves:
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 1.0 * P or bh < 0.5 * P:
            continue
        lines = ocr_block(norm, rules, box, P)
        lines = [l for l in lines if any(re.search(rf"[{GEO}0-9]", w[0]) for w in l["words"])]
        verify_marks(lines, ink, P)
        if not lines:
            continue
        blocks.append(dict(box=tuple(int(v) for v in box), lines=lines))
    drop_garbage(blocks)
    mark_headers(blocks, hr, P, H)
    if debug_dir:
        save_debug(norm, blocks, vr, hr, os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{pno + 1:03d}.jpg"))
    return dict(blocks=blocks, P=P, size=(W, H))


def _small_marks(ink, x0, y0, x1, y1, P, max_h):
    """[x0,x1]x[y0,y1] არეში პატარა ცალკე მდგომი ნიშნები (წერტილი, მძიმე, ბრჭყალი) და
    დანარჩენი (ასოების) მელნის საზღვრები."""
    H, W = ink.shape
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return [], None, None
    n, _, st, cen = cv2.connectedComponentsWithStats(ink[y0:y1, x0:x1], connectivity=8)
    marks, big_l, big_r = [], None, None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 2 and a <= 0.06 * P * P and w <= 0.4 * P and h <= max_h:
            marks.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        elif a > 0.06 * P * P or h > max_h:
            big_l = x0 + x if big_l is None else min(big_l, x0 + x)
            big_r = x0 + x + w if big_r is None else max(big_r, x0 + x + w)
    return marks, big_l, big_r


def verify_marks(lines, ink, P):
    """Tesseract ხშირად "იგონებს" წერტილს სტრიქონის ბოლოს და ბრჭყალს ლაქის ადგილას.
    ვამოწმებთ, ნამდვილად არის თუ არა სურათზე შესაბამისი ნიშანი სიტყვის ბოლოს/დასაწყისში:
    წერტილი - პატარა ლაქა ფუძის ხაზთან; ბრჭყალი - ორი ნიშანი ან ერთი საკმაოდ დიდი
    (ერთი წვრილი ლაქა ქაღალდის ნაგავია)."""
    dot = max(2.0, 0.08 * P)

    def is_quote(ms):
        dims = [(m[2] - m[0], m[3] - m[1]) for m in ms]
        return (any(min(d) >= 0.18 * P and max(d) >= 0.22 * P for d in dims) or
                sum(max(d) >= 0.15 * P for d in dims) >= 2)

    for l in lines:
        ws = []
        for w in l["words"]:
            t, conf, x0, y0, x1, y1 = w
            if len(t) > 1 and re.search(rf"[{GEO}]", t[-2]) and t[-1] in ".,:;“”\"'":
                marks, _, big_r = _small_marks(ink, x1 - 0.8 * P, y0 - 0.2 * P, x1 + 0.3 * P, y1 + 0.2 * P, P, 0.5 * P)
                marks = [m for m in marks if big_r is None or m[0] >= big_r - 1]
                low = [m for m in marks if m[1] > y0 + 0.35 * (y1 - y0) and min(m[2] - m[0], m[3] - m[1]) >= dot]
                hyphen = any(m[2] - m[0] >= 0.22 * P and m[2] - m[0] >= 1.8 * (m[3] - m[1]) for m in marks)
                if t[-1] == ".":
                    ok = any(m[3] - m[1] <= 2.0 * (m[2] - m[0]) for m in low)
                elif t[-1] == ",":
                    ok = bool(low)
                elif t[-1] in ":;":
                    ok = sum(min(m[2] - m[0], m[3] - m[1]) >= dot for m in marks) >= 2
                else:
                    ok = is_quote(marks)
                if hyphen and not ok:
                    t = t[:-1] + "-"                  # დეფისი, რომელიც "."-დ ან ბრჭყალად წაიკითხა
                elif not ok:
                    t = t[:-1]
            if len(t) > 1 and t[0] in "„“\"'«" and re.search(rf"[{GEO}]", t[1]):
                marks, big_l, _ = _small_marks(ink, x0 - 0.3 * P, y0 - 0.2 * P, x0 + 0.8 * P, y1, P, 0.5 * P)
                if not is_quote([m for m in marks if big_l is None or m[2] <= big_l + 1]):
                    t = t[1:]
            ws.append((t, conf, x0, y0, x1, y1))
        l["words"] = ws


def _line_height(l):
    wh = [w[5] - w[3] for w in l["words"] if len(w[0]) >= 2]
    return float(np.median(wh)) if wh else 0.0


def drop_garbage(blocks):
    """გაზეთის სახელი დეკორატიული შრიფტით (ძალიან დიდი ასოები) და ცალკე მდგომი 1-2 სტრიქონი,
    რომელშიც თითქმის არაფერი ამოიცნო (ორნამენტი, შტამპი, ლაქა). ჩვეულებრივ ტექსტს არ ეხება."""
    hs = [_line_height(l) for b in blocks for l in b["lines"]]
    hs = [h for h in hs if h > 0]
    if not hs:
        return
    body = float(np.median(hs))
    lex = lexicon()
    for b in blocks:
        short = len(b["lines"]) <= 2
        keep = []
        for l in b["lines"]:
            ws = l["words"]
            n = sum(len(w[0]) for w in ws)
            conf = sum(max(0.0, w[1]) * len(w[0]) for w in ws) / max(1, n)
            if _line_height(l) >= 2.5 * body and conf < 90:
                continue
            known = sum(word_core(w[0]) in lex for w in ws if len(word_core(w[0])) >= 3)
            if short and line_score(l) < 60 and known <= 3:
                continue
            keep.append(l)
        b["lines"] = keep
    blocks[:] = [b for b in blocks if b["lines"]]


def mark_headers(blocks, hr, P, H):
    """გვერდის ზედა სტრიქონი (კოლონტიტული): ზემოთ, ერთ სტრიქონში, ქვეშ ხაზი აქვს."""
    for b in blocks:
        x0, y0, x1, y1 = b["box"]
        b["header"] = False
        if y1 > 0.12 * H or len(b["lines"]) > 1:
            continue
        below = hr[y1:min(H, int(y1 + 2 * P)), :]
        if below.any():
            b["header"] = True


# ---------------------------- ტექსტის აწყობა ----------------------------


_BAD = re.compile(r"[^\u10D0-\u10FF\u0400-\u04FFA-Za-z0-9\s.,;:!?()\[\]\"'«»„“”‘’\-–—%/№§*+=<>&]")
_JUNK_TOKEN = re.compile(r"^[\"'“”„«»_.,:;·•`´‘’^~*|\\/<>]+$")


def clean_word(w):
    w = "".join(chr(ord(c) - 0x1C90 + 0x10D0) if 0x1C90 <= ord(c) <= 0x1CBA else c for c in w)
    w = w.replace("|", "").replace("_", "")
    w = _BAD.sub("", w)
    w = re.sub(r"[-–—]{2,}", "—", w)
    w = re.sub(r"^['`´‘’^]+", "", w)
    w = re.sub(rf"(?<=[{GEO}])['`´‘’^]+$", "", w)
    return w


def line_tokens(line):
    toks = [re.sub(rf"(?<=[{GEO}]{{2}})\.(?=[{GEO}])", "", clean_word(w[0])) for w in line["words"]]
    toks = [t for t in toks if t and not _JUNK_TOKEN.match(t)]
    return toks


class Lexicon:
    """სიტყვების სია + სიხშირეები (უკვე ამოკითხული ტექსტებიდან და მიმდინარე დოკუმენტიდან)."""

    def __init__(self, words, freq=None):
        self.words, self.freq = words, freq or {}

    def __contains__(self, w):
        return w in self.words or self.freq.get(w, 0) >= 3

    def f(self, w):
        return self.freq.get(w, 0)


_QUOTEISH = "\"'“”„‘’`´"


def try_join(a, b, at_margin, lex, detached=False):
    """a - სტრიქონის ბოლო სიტყვა, b - შემდეგის პირველი. აბრუნებს გაერთიანებულს ან None.
    detached=True: დეფისი ცალკე სიტყვად წაიკითხა ("სიტყ -")."""
    m = re.match(rf"^(.*[{GEO}])([^{GEO}]*)$", a)
    if not m or re.search(r"[0-9A-Za-z\u0400-\u04FF]", m.group(2)):
        return None
    a_core_full, tail = m.group(1), m.group(2).strip()
    a_core = re.search(rf"[{GEO}]+$", a_core_full).group(0)
    bm = re.match(rf"^[{_QUOTEISH}]*([{GEO}]+)(.*)$", b)
    if not bm:
        return None
    b_core, b_rest = bm.group(1), bm.group(2)
    w = a_core + b_core
    kw, ka, kb = w in lex, a_core in lex, b_core in lex
    joined = a_core_full + b_core + b_rest
    if tail and tail[0] in HYPHENS + DASHES:
        if detached and not kw and (ka or kb):
            return None                                             # ალბათ ნამდვილი ტირეა
        if not kw and ka and kb and len(a_core) >= 4 and len(b_core) >= 4:
            return a_core_full + "-" + b_core + b_rest              # შედგენილი სიტყვა: ზრდა-განვითარება
        return joined
    if not at_margin:
        return None
    # დეფისი დაიკარგა ან "."-დ / ბრჭყალად წაიკითხა
    if all(c in _QUOTEISH for c in tail):
        if kw and (tail or not (ka and kb) or plausible_join(a_core, b_core, w, lex)):
            return joined
        if tail and not ka and not kb and len(a_core) >= 2:
            return joined
    elif all(c in ".,:;" for c in tail):
        if kw and not (ka and kb):
            return joined
    return None


def plausible_join(a, b, w, lex):
    """დეფისის გარეშე: ორივე ნაწილი ცალკეც სიტყვაა ("და" + "სხვა") - ვაერთებთ მხოლოდ თუ
    გაერთიანებული სიტყვა ტექსტებში ხშირად გვხვდება."""
    fw = lex.f(w)
    return fw >= 2 and fw >= 0.05 * min(lex.f(a), lex.f(b))


def split_paragraphs(block, P):
    """ბლოკის სტრიქონები -> [(kind, [lines])], kind = 'head' | 'para'."""
    lines = block["lines"]
    if not lines:
        return []
    x0s = np.array([l["box"][0] for l in lines])
    x1s = np.array([l["box"][2] for l in lines])
    left = float(np.percentile(x0s, 20))
    right = float(np.percentile(x1s, 80))
    width = max(1.0, right - left)
    hs = []
    for l in lines:
        wh = [w[5] - w[3] for w in l["words"] if len(w[0]) >= 2]
        hs.append(float(np.median(wh)) if wh else 0.0)
    body_h = float(np.median([h for h in hs if h > 0])) if any(hs) else P
    items = []
    for i, l in enumerate(lines):
        lg, rg = l["box"][0] - left, right - l["box"][2]
        big = hs[i] >= 1.35 * body_h and len(lines) > 1
        centered = lg > 1.5 * P and rg > 1.5 * P and abs(lg - rg) < max(2.0 * P, 0.25 * width)
        l["at_margin"] = rg < 1.2 * P
        l["indent"] = lg > 0.8 * P
        items.append(("head" if (big or (centered and len(lines) > 2)) else "body", l))
    paras = []
    for i, (kind, l) in enumerate(items):
        if kind == "head":
            if paras and paras[-1][0] == "head" and items[i - 1][0] == "head":
                paras[-1][1].append(l)
            else:
                paras.append(["head", [l]])
            continue
        prev = items[i - 1] if i else None
        new = prev is None or prev[0] == "head" or l["indent"] or not prev[1]["at_margin"]
        if new or not paras or paras[-1][0] != "para":
            paras.append(["para", [l]])
        else:
            paras[-1][1].append(l)
    return paras


def join_paragraph(lines, lex):
    """სტრიქონებს აერთებს ერთ ტექსტად, გაწყვეტილ სიტყვებს აღადგენს."""
    words, prev_margin = [], None
    for l in lines:
        t = line_tokens(l)
        if not t:
            continue
        if words and prev_margin is not None and join_words(words, t[0], prev_margin, lex):
            t = t[1:]
        words.extend(t)
        prev_margin = l.get("at_margin", True)
    return words


def join_words(words, first, at_margin, lex):
    """words-ის ბოლო სიტყვას (ადგილზე) უერთებს first-ს, თუ ეს გაწყვეტილი სიტყვაა."""
    if re.fullmatch(rf"[{HYPHENS}{DASHES}]+", words[-1]) and len(words) >= 2:
        j = try_join(words[-2] + words[-1], first, at_margin, lex, detached=True)
        if j is not None:
            words[-2:] = [j]
            return True
        return False
    j = try_join(words[-1], first, at_margin, lex)
    if j is not None:
        words[-1] = j
        return True
    return False


def fix_quotes(words):
    """დაუწყვილებელი ბრჭყალები (ლაქებისგან) - იშლება."""
    out = list(words)
    for i, w in enumerate(out):
        if w and w[0] in OPEN_Q:
            window = " ".join(out[i:i + 40])
            if not re.search(r"[”“»\"]", window[1:]):
                out[i] = w[1:]
        if w and w[-1] in CLOSE_Q + "“":
            window = " ".join(out[max(0, i - 40):i + 1])
            if not re.search(r"[„«\"“]", window[:-1]):
                out[i] = out[i][:-1]
    return [w for w in out if w]


_END = re.compile(r"[.!?:;»”“\")]$")


def assemble_document(pages, lex):
    """გვერდები -> ტექსტი. აბზაცები სვეტიდან სვეტზე და გვერდიდან გვერდზე გრძელდება."""
    out_pages = []
    carry = None          # წინა ბლოკის/გვერდის ბოლო აბზაცი, თუ ღიაა: (page_idx, para_idx, margin)
    for pi, page in enumerate(pages):
        paras = []
        P = page["P"]
        for b in page["blocks"]:
            if b.get("header") and DROP_PAGE_HEADERS:
                continue
            if KEEP_LINES:
                paras.append(("para", [" ".join(line_tokens(l)) for l in b["lines"]]))
                continue
            for kind, lines in split_paragraphs(b, P):
                if kind == "head":
                    for l in lines:
                        l["at_margin"] = True       # სათაური შუაშია - დეფისის დაკარგვაც შესაძლებელია
                words = join_paragraph(lines, lex)
                if not words:
                    continue
                first = lines[0]
                if kind == "para" and carry is not None and not first.get("indent"):
                    cpi, cpar, cmargin = carry
                    target = paras if cpi == pi else out_pages[cpi]
                    tw = target[cpar][1]
                    if tw and join_words(tw, words[0], cmargin, lex):
                        words = words[1:]
                        if cpi == pi:
                            tw.extend(words)
                            words = []
                    elif cpi == pi and not _END.search(tw[-1]):
                        tw.extend(words)
                        words = []
                if words:
                    paras.append((kind, words))
                last = lines[-1]
                if kind == "para":
                    open_ = last.get("at_margin") and not _END.search(paras[-1][1][-1] if paras[-1][1] else ".")
                    carry = (pi, len(paras) - 1, True) if open_ else None
                else:
                    carry = None
        out_pages.append(paras)
    texts = []
    for paras in out_pages:
        chunks = []
        for kind, words in paras:
            if not words:
                continue
            if KEEP_LINES:
                chunks.append("\n".join(words))
                continue
            t = " ".join(fix_quotes(words))
            t = re.sub(r"\s+([,.;:!?])", r"\1", t)
            if kind == "head" and MARK_HEADINGS:
                t = "## " + t
            chunks.append(t)
        texts.append("\n\n".join(chunks))
    return texts


# ---------------------------- debug ----------------------------


def save_debug(norm, blocks, vr, hr, path):
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    vis[(vr | hr) > 0] = (60, 200, 60)
    th = max(1, norm.shape[1] // 600)
    for k, b in enumerate(blocks):
        x0, y0, x1, y1 = b["box"]
        col = (160, 160, 160) if b.get("header") else (0, 0, 230)
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2 * th)
        for l in b["lines"]:
            lx0, ly0, lx1, ly1 = [int(v) for v in l["box"]]
            cv2.rectangle(vis, (lx0, ly0), (lx1, ly1), (230, 150, 0), 1)
        lab = str(k + 1)
        (tw, tg), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * th, th)
        cv2.rectangle(vis, (x0, y0), (x0 + tw + 8, y0 + tg + 10), col, -1)
        cv2.putText(vis, lab, (x0 + 4, y0 + tg + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * th, (255, 255, 255), th)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------- PDF ----------------------------


def corpus_freq(folder):
    """სიტყვების სიხშირე უკვე ამოკითხულ ტექსტებში. 3-ჯერ მაინც შეხვედრილი სიტყვა ლექსიკონს
    ემატება (ძველი მართლწერა: "ერთიანათ", "იმდენათ" და ა.შ.)."""
    c = Counter()
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(".txt"):
                with open(os.path.join(folder, f), encoding="utf-8") as fh:
                    c.update(re.findall(rf"[{GEO}]{{2,}}", fh.read()))
    return c


def document_freq(pages):
    c = Counter()
    for page in pages:
        for b in page["blocks"]:
            for l in b["lines"]:
                for w in l["words"]:
                    c.update(re.findall(rf"[{GEO}]{{2,}}", w[0]))
    return c


def parse_pages(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        a = int(a)
        b = int(b) if b else a
        out.extend(range(a - 1, min(b, n)))
    return out


def run(pdfs, out_dir, pages_spec=None, debug=False):
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    freq = corpus_freq(out_dir)
    if not lexicon():
        print(f"გაფრთხილება: {WORDLIST} ვერ მოიძებნა - გაწყვეტილი სიტყვები მხოლოდ დეფისით აღდგება.")

    jobs, counts = [], {}
    for p in pdfs:
        n = len(fitz.open(p))
        pg = parse_pages(pages_spec, n)
        counts[p] = pg
        jobs += [(p, i, debug_dir) for i in pg]
    results = {}
    t0 = time.time()
    done = {p: 0 for p in pdfs}
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(_settings(),)) as ex:
        futs = {ex.submit(process_page, j): j for j in jobs}
        for fut in as_completed(futs):
            p, i, _ = futs[fut]
            try:
                results[(p, i)] = fut.result()
            except Exception as e:
                print(f"   შეცდომა: {os.path.basename(p)} გვერდი {i + 1}: {e}", flush=True)
                results[(p, i)] = dict(blocks=[], P=30.0, size=(0, 0), error=str(e))
            done[p] += 1
            if done[p] == len(counts[p]):
                write_pdf(p, [results.pop((p, k)) for k in counts[p]], counts[p], out_dir, freq)
                print(f"-> მზადაა: {os.path.basename(p)} ({time.time() - t0:.0f} წმ)", flush=True)


def write_pdf(p, pages, pnos, out_dir, freq):
    lex = Lexicon(lexicon(), freq + document_freq(pages))
    texts = assemble_document(pages, lex)
    base = os.path.splitext(os.path.basename(p))[0]
    out_path = os.path.join(out_dir, base + ".txt")
    parts = []
    for pno, page, t in zip(pnos, pages, texts):
        err = f"[OCR შეცდომა: {page['error']}]\n" if page.get("error") else ""
        parts.append(f"===== გვერდი {pno + 1} =====\n\n{err}{t}\n\n")
    tmp = out_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(parts)
    os.replace(tmp, out_path)


_SETTING_NAMES = ["TESSERACT_CMD", "TESSDATA_DIR", "LANG", "WORDLIST", "QUALITY", "MIN_LINE_PITCH",
                  "MAX_LINE_PITCH", "TARGET_LINE_PITCH"]


def _settings():
    return {k: globals()[k] for k in _SETTING_NAMES}


def _init_worker(settings):
    globals().update(settings)
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    use_tessdata()


def check_tesseract():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    use_tessdata()
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception as e:
        sys.exit(f"Tesseract ვერ გაეშვა ({e}). შეამოწმეთ TESSERACT_CMD გზა.")
    missing = [l for l in LANG.split("+") if l not in langs]
    if missing:
        sys.exit(f"Tesseract-ში არ არის ენა/ენები: {', '.join(missing)}.\n"
                 f"ჩამოტვირთეთ <ენა>.traineddata და ჩააგდეთ tessdata საქაღალდეში.\n"
                 f"ახლა ხელმისაწვდომია: {', '.join(sorted(langs))}")


def main():
    global TESSERACT_CMD, TESSDATA_DIR, LANG, QUALITY, WORKERS, DROP_PAGE_HEADERS, MARK_HEADINGS, KEEP_LINES
    ap = argparse.ArgumentParser(description="ძველი ქართული გაზეთების OCR, ვერსია 6")
    ap.add_argument("inputs", nargs="*", help="PDF ფაილები ან საქაღალდეები (ნაგულისხმევი: INPUT_PDF_FOLDER)")
    ap.add_argument("--out", default=OUTPUT_TXT_FOLDER)
    ap.add_argument("--pages", help="მაგ. 1-3,5")
    ap.add_argument("--debug", action="store_true", default=DEBUG_IMAGES)
    ap.add_argument("--overwrite", action="store_true", help="უკვე არსებულ TXT-ებსაც თავიდან ამუშავებს")
    ap.add_argument("--fast", action="store_true", help="ერთი ვარიანტი (QUALITY=fast)")
    ap.add_argument("--tesseract", default=TESSERACT_CMD)
    ap.add_argument("--tessdata", default=TESSDATA_DIR)
    ap.add_argument("--lang", default=LANG)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--keep-headers", action="store_true", help="გვერდის ზედა სტრიქონი ტექსტში დარჩეს")
    ap.add_argument("--mark-headings", action="store_true", default=MARK_HEADINGS)
    ap.add_argument("--lines", action="store_true", default=KEEP_LINES, help="სტრიქონები ცალ-ცალკე")
    a = ap.parse_args()
    TESSERACT_CMD, TESSDATA_DIR, LANG, WORKERS = a.tesseract, a.tessdata, a.lang, a.workers
    if a.fast:
        QUALITY = "fast"
    if a.keep_headers:
        DROP_PAGE_HEADERS = False
    MARK_HEADINGS, KEEP_LINES = a.mark_headings, a.lines
    check_tesseract()

    pdfs = []
    for inp in (a.inputs or [INPUT_PDF_FOLDER]):
        if os.path.isdir(inp):
            for root, _d, files in os.walk(inp):
                pdfs += [os.path.join(root, f) for f in files if f.lower().endswith(".pdf")]
        elif inp.lower().endswith(".pdf"):
            pdfs.append(inp)
    pdfs.sort()
    if not pdfs:
        print("PDF ფაილები ვერ მოიძებნა.")
        return
    todo = []
    for p in pdfs:
        txt = os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if SKIP_EXISTING and not a.overwrite and not a.pages and os.path.exists(txt):
            print(f"გამოტოვებულია (უკვე არსებობს): {os.path.basename(p)}")
            continue
        todo.append(p)
    print(f"\nსულ {len(pdfs)} PDF, დასამუშავებელი {len(todo)}. პარალელურად: {WORKERS} გვერდი\n")
    if todo:
        run(todo, a.out, a.pages, a.debug)
    print(f"\nდასრულდა. შედეგები: {a.out}")


if __name__ == "__main__":
    main()
