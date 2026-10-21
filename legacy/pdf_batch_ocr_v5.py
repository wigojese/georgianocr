# -*- coding: utf-8 -*-
"""
ძველი ქართული გაზეთების / ჟურნალების PDF-ების OCR (Tesseract, ქართული) - ვერსია 4.

რას აკეთებს:
  1. გვერდს იღებს სკანის საკუთარი რეზოლუციით (PDF-ის "გვერდის ზომას" არ ენდობა)
  2. ზომავს ასოს სიმაღლეს და სურათს ისე ადიდებს/აპატარავებს, რომ Tesseract-ისთვის
     საუკეთესო ზომა იყოს (დაბალი ხარისხის სკანებზე ეს ძალიან მნიშვნელოვანია)
  3. ასწორებს არათანაბარ, გაყვითლებულ, ლაქიან ფონს (Sauvola ბინარიზაცია)
  4. პოულობს სვეტების გამყოფ ხაზებს (მრუდე და წყვეტილ ხაზებსაც) და იყენებს მათ
     საზღვრებად, შემდეგ შლის, რომ ტექსტში "|" არ გაჩნდეს
  5. ტექსტს ყოფს ბლოკებად (აბზაცები, სათაურები, სვეტები) და ალაგებს სწორი
     თანმიმდევრობით, მათ შორის რამდენიმე სვეტზე გადაჭიმული სათაურების შემთხვევაში
  6. პატარა ბლოკებს აერთიანებს სტატიებად (სათაური + მისი ტექსტი, ჩარჩოში მოქცეული
     განცხადება და ა.შ.), სანამ გვერდზე MAX_BLOCKS ნაწილი არ დარჩება
  7. ტექსტს ბლოკში სვეტ-სვეტად კითხულობს და ასუფთავებს შედეგს ("სიტყ-\\nვა" -> "სიტყვა", ხაზების ნარჩენები და ა.შ.)
  8. DEBUG_IMAGES=True-ზე ინახავს სურათებს: წითელი = ბლოკები და მათი რიგი, მწვანე = ნაპოვნი ხაზები

გაშვება:  python pdf_batch_ocr_v4.py
"""

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================ პარამეტრები ============================

# Tesseract-ის გზა
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_PDF_FOLDER = "pdf_files"
OUTPUT_TXT_FOLDER = "extracted_texts"

# ენა. 1910-იანი წლების გაზეთებისთვის "kat" საუკეთესოა.
# "kat+kat_old" მხოლოდ მაშინ, თუ ტექსტში ხშირად გვხვდება ჱ, ჲ, ჳ, ჴ, ჵ (XIX საუკუნე).
# თუ ქართულ გაზეთში ბევრი რუსული ჩანართია: "kat+rus"
LANG = "kat"

# რუსული (და სხვა) პრესისთვის ენა შეირჩევა ქვესაქაღალდით: pdf_files\ru\გაზეთი.pdf
# ქვესაქაღალდის სახელი -> Tesseract-ის ენა. უშუალოდ pdf_files-ში მდებარე ფაილები LANG-ს იყენებს.
# ძველი ორთოგრაფიის რუსულისთვის (ѣ, і, ѳ) ისევ "rus" გამოიყენება.
LANG_BY_FOLDER = {"ru": "rus", "ka": "kat", "ka_ru": "kat+rus"}

# ასოს სასურველი სიმაღლე პიქსელებში. სკრიპტი ყველა გვერდს ამ ზომაზე მოარგებს.
# DPI-ს ცვლა აღარ არის საჭირო.
TARGET_CHAR_HEIGHT = 28

# როგორ დაიყოს გვერდი:
#   "blocks"    - ამ სკრიპტის სეგმენტაცია (გამყოფი ხაზების მქონე გაზეთებისთვის საუკეთესო)
#   "tesseract" - Tesseract-ის ჩაშენებული ანალიზი (სცადეთ, თუ "blocks" რომელიმე გაზეთზე ცუდად მუშაობს)
LAYOUT = "blocks"

# "best" - თითოეულ ბლოკს ორნაირად კითხულობს და უკეთესს ირჩევს (≈2-ჯერ ნელი, უფრო ზუსტი)
# "fast" - ერთხელ კითხულობს
QUALITY = "best"

# ბლოკი, რომელშიც Tesseract-ის სანდოობა ამაზე დაბალია, ნაგვად ითვლება და გამოიტოვება
# (ლაქები, ორნამენტები, სურათები). თუ ტექსტი იკარგება -> შეამცირეთ (25)
MIN_CONFIDENCE = 40

# გვერდზე მაქსიმუმ რამდენი ნაწილი (სტატია / განცხადება / სათაური) დარჩეს.
# 0 = ლიმიტის გარეშე (თითოეული სვეტი ცალკე). ბევრი ნაწილი გამოდის -> შეამცირეთ, ზედმეტად ერთიანდება -> გაზარდეთ
MAX_BLOCKS = 20
HEAD_RATIO = 1.45       # რამდენჯერ დიდი ასოები ითვლება სათაურად (ჩვეულებრივ ტექსტთან შედარებით)
TEXT_GAP = 2.5          # ერთი სვეტის ორი ტექსტი რა დაშორებამდე ითვლება ერთ სტატიად (ასოს სიმაღლეებში)
SIDE_INK_MAX = 0.30
VFILTER_MAX_COMPS = 120
PITCH_RATIO_MAX = 2.6        # სტრიქონის ბიჯი / ასოს სიმაღლე ამაზე მეტია -> ასო არასწორად ფასდება
PITCH_RATIO_TYPICAL = 2.0
DEBUG_RULES = False   # ხაზის გვერდებზე მელნის მაქს. სიმკვრივე (ტექსტში გამავალი "ხაზების" გასაფილტრად)

DEBUG_IMAGES = True     # დაჭრის სქემა: extracted_texts/_debug/
SKIP_EXISTING = True    # უკვე დამუშავებულ PDF-ებს გამოტოვებს
WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))   # რამდენი PDF ერთდროულად (დიდი სკანები ბევრ მეხსიერებას მოითხოვს)

# ბლოკების აწყობა (ერთეული = ასოს სიმაღლე)
WORD_JOIN = 0.8         # სიტყვებს შორის რა დაშორებამდე აერთიანებს. სვეტები ერთმანეთს ეწებება -> შეამცირეთ (0.6)
LINE_JOIN = 1.0         # სტრიქონებს შორის რა დაშორებამდე აერთიანებს ერთ ბლოკად

JOIN_HYPHENS = True     # "სიტყ-\nვა" -> "სიტყვა"
MERGE_LINES = True      # ბლოკის სტრიქონებს აერთებს ერთ აბზაცად
CLEAN_SYMBOLS = True    # შლის ხაზების ნარჩენებს, ასომთავრულს/ნუსხურს და უცნაურ სიმბოლოებს

# =====================================================================

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # Tesseract-ის ნაკადები ერთმანეთს არ უნდა შეეჯახოს

import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

GEO = "\u10D0-\u10FF"
MAX_PIXELS = 60_000_000   # ზედმეტად დიდი სურათი მეხსიერებას გადაავსებს


# ---------------------------- გვერდის ჩატვირთვა ----------------------------

def load_page(doc, page):
    """გვერდი სკანის საკუთარი რეზოლუციით (ან 300 DPI, თუ სკანი არ არის)."""
    zoom = 300 / 72
    imgs = page.get_images(full=True)
    if imgs:
        widths = [im[2] for im in imgs]
        zoom = max(widths) / max(1.0, page.rect.width)
    zoom = min(zoom, (MAX_PIXELS / max(1.0, page.rect.width * page.rect.height)) ** 0.5)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


# ---------------------------- ბინარიზაცია ----------------------------

def normalize_background(gray):
    """ფონს ათანაბრებს: ქაღალდი თეთრდება, ლაქები და ჩრდილები სუსტდება."""
    h, w = gray.shape
    f = 8
    small = cv2.resize(gray, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    norm = gray.astype(np.float32) / np.maximum(bg, 1) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def sauvola(gray, win, k=0.18, R=128.0):
    """Sauvola ბინარიზაცია - ძველი, დაზიანებული დოკუმენტებისთვის ერთ-ერთი საუკეთესო მეთოდი.
    აბრუნებს ink ნიღაბს (255 = მელანი)."""
    win = max(15, int(win)) | 1
    g = gray.astype(np.float32)
    m = cv2.boxFilter(g, cv2.CV_32F, (win, win))
    s = cv2.boxFilter(g * g, cv2.CV_32F, (win, win))
    s = np.sqrt(np.maximum(s - m * m, 0))
    t = m * (1 + k * (s / R - 1))
    del s, m
    return ((g < t) & (g < 200)).astype(np.uint8) * 255


def remove_small(mask, min_area):
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def remove_border_touching(ink, margin=3):
    """სკანის კიდეზე მიბჯენილი ყველაფერი (შავი ზოლები, ფურცლის კიდე, ჩრდილი)."""
    h, w = ink.shape
    n, labels, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    x, y = st[:, cv2.CC_STAT_LEFT], st[:, cv2.CC_STAT_TOP]
    x2, y2 = x + st[:, cv2.CC_STAT_WIDTH], y + st[:, cv2.CC_STAT_HEIGHT]
    touch = (x <= margin) | (y <= margin) | (x2 >= w - margin) | (y2 >= h - margin)
    touch[0] = False
    out = ink.copy()
    out[touch[labels]] = 0
    return out


def _median_component_height(ink):
    H = ink.shape[0]
    _, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs, ws, ar = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_AREA]
    m = (hs >= 4) & (hs <= H / 25) & (ar >= 10)
    if m.sum() < 50:
        return None
    # წვრილი ლაქები (ქაღალდის ტექსტურა, მეორე მხრიდან გამოსული ტექსტი) ასოებზე ბევრად
    # მეტია და მედიანას "ჩამოსწევდა" - მათ ვაცლით
    ref = np.percentile(hs[m], 90)
    m &= hs >= 0.35 * ref
    return float(np.median(hs[m]))


def estimate_char_height(ink):
    """ასოს სიმაღლე. ძველ ბეჭდვაში ასოები ხშირად "დაფლეთილია", ამიტომ ცალკე ნაწილებს
    კი არ ვზომავთ, არამედ სიტყვებს (ასოებს ჰორიზონტალურად ვაწებებთ) - ეს გაცილებით სტაბილურია."""
    h0 = _median_component_height(ink)
    if h0 is None:
        return 20.0
    k = max(3, int(0.6 * h0)) | 1
    words = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    hw = _median_component_height(words)
    if hw is None:
        return h0
    return 0.65 * hw     # სიტყვის სიმაღლე (ზედა და ქვედა "კუდებით") ≈ 1.5 x ასოს სიმაღლე


def deskew(gray):
    """მთლიანი გვერდის დახრა (-3°..+3°) სტრიქონების პროექციით."""
    h, w = gray.shape
    scale = 1200.0 / w
    small = cv2.resize(gray, (1200, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sh, sw = b.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-3.0, 3.01, 0.2):
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        r = cv2.warpAffine(b, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = angle, score
    if abs(best_angle) < 0.1:
        return gray
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def line_pitch(ink, ch):
    """სტრიქონებს შორის მანძილი (ავტოკორელაცია სვეტის ზოლებზე). მკრთალ სკანზე ასოები
    გატეხილია და ch-ს ამცირებს - სტრიქონის ბიჯი ასეთ დროს სანდოა."""
    H, W = ink.shape
    ks = []
    for x0 in range(0, max(1, W - W // 6), max(1, W // 8)):
        v = (ink[:, x0:x0 + W // 6] > 0).sum(1).astype(np.float32)
        v = cv2.GaussianBlur(v.reshape(-1, 1), (1, 0), max(1.0, 0.1 * ch)).ravel()
        v -= v.mean()
        ac = np.correlate(v, v, "full")[len(v) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = max(4, int(0.8 * ch)), min(len(ac) - 1, int(8 * ch))
        if hi <= lo:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        ks.append((float(ac[k]), k))
    ks.sort(reverse=True)
    return float(np.median([k for _, k in ks[:4]])) if ks else 0.0


def fix_char_height(ink, ch):
    p = line_pitch(ink, ch)
    if p and p / ch > PITCH_RATIO_MAX:
        return p / PITCH_RATIO_TYPICAL
    return ch


def prepare_image(gray):
    """აბრუნებს: სასურველ ზომაზე მორგებულ ნაცრისფერ სურათს, ink ნიღაბს და ასოს სიმაღლეს."""
    norm = normalize_background(gray)
    ink = remove_border_touching(sauvola(norm, 41))
    ch = fix_char_height(ink, estimate_char_height(ink))

    scale = TARGET_CHAR_HEIGHT / ch
    scale = min(scale, (MAX_PIXELS / (gray.shape[0] * gray.shape[1])) ** 0.5)
    scale = max(0.4, min(3.0, scale))
    if not 0.9 <= scale <= 1.1:
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        norm = cv2.resize(norm, None, fx=scale, fy=scale, interpolation=interp)
    norm = deskew(norm)

    ink = sauvola(norm, 2.2 * TARGET_CHAR_HEIGHT)
    ink = remove_border_touching(ink)
    ink = cv2.medianBlur(ink, 3)                       # წვრილი "ქვიშა"
    ch = fix_char_height(ink, estimate_char_height(ink))
    return norm, ink, ch


# ---------------------------- გამყოფი ხაზები ----------------------------

def find_rules(ink, ch):
    """ვერტიკალური და ჰორიზონტალური გამყოფი ხაზები. ოდნავ დახრილ/მრუდე და
    წყვეტილ ხაზებსაც პოულობს. აბრუნებს (vrules, hrules, frames) ნიღბებს."""
    # ვერტიკალური: ჯერ ოდნავ ვასქელებთ, რომ დახრილი ხაზიც "სწორ" ზოლად დაითვალოს
    t = max(3, int(0.15 * ch)) | 1
    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (t, 1)))
    v = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(3 * ch))))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(2 * ch))))
    v = _keep_long(v, axis=0, min_len=6 * ch, max_thick=1.0 * ch)

    thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, t)))
    hz = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(5 * ch), 1)))
    hz = cv2.morphologyEx(hz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(2 * ch), 1)))
    hz = _keep_long(hz, axis=1, min_len=6 * ch, max_thick=1.0 * ch)

    # ძალიან მკრთალი, "წერტილოვანი" ხაზები: ცალკეული ვიწრო, მაღალი ნაჭრები, რომლებიც
    # ერთმანეთის ქვემოთ დგას - ვაერთებთ და თუ გრძელი ხაზი გამოვიდა, ესეც გამყოფია
    n, labels, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    fw, fh = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    frag = (fw <= max(2, 0.25 * ch)) & (fh >= 0.8 * ch) & (fh <= 30 * ch)
    frag[0] = False
    f = np.where(frag[labels], 255, 0).astype(np.uint8)
    f = cv2.dilate(f, cv2.getStructuringElement(cv2.MORPH_RECT, (t, 1)))
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(3 * ch))))
    f = _keep_long(f, axis=0, min_len=6 * ch, max_thick=1.0 * ch)

    # ჩარჩოები და მრუდე ხაზები: ძალიან დიდი, მაგრამ "ცარიელი" ობიექტები (ასო ასეთი არ არის)
    frame = (((fw > 12 * ch) | (fh > 12 * ch)) &
             (st[:, cv2.CC_STAT_AREA] < 0.12 * fw * fh) &
             (st[:, cv2.CC_STAT_AREA] < 1.2 * ch * (fw + fh)))
    frame[0] = False
    fr = np.where(frame[labels], 255, 0).astype(np.uint8)

    return v | f | fr | _vline_filter(ink, ch), hz, fr


def _shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    out[yd, xd] = a[ys, xs]
    return out


def _vline_filter(ink, ch):
    """"ვიწრო მუქი ვერტიკალური ზოლი, ორივე მხარეს თეთრი" - პოულობს დაფლეთილ,
    წერტილოვან, მკრთალ გამყოფ ხაზებს, რომლებსაც ჩვეულებრივი მეთოდი ვერ ხედავს.
    (ჰორიზონტალურად არ გამოდგება: ქართული სტრიქონი თავადაც "ზოლს" ჰგავს.)"""
    s = 0.5
    d = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    c = ch * s
    L = max(5, int(4 * c)) | 1
    cw = max(3, int(0.25 * c)) | 1
    sw = max(3, int(0.3 * c)) | 1
    off = max(cw // 2 + sw // 2 + 1, int(0.5 * c))
    center = cv2.boxFilter(d, -1, (cw, L))
    side = cv2.boxFilter(d, -1, (sw, L))
    s1, s2 = _shift(side, off, 0), _shift(side, -off, 0)
    nb = int(1.5 * cw)
    n1, n2 = _shift(center, nb, 0), _shift(center, -nb, 0)     # მეზობლები
    m = ((center >= 0.3) & (center - s1 >= 0.18) & (center - s2 >= 0.18) &
         (center - n1 >= 0.12) & (center - n2 >= 0.12)            # ვიწროა (დიდი ასოს შტრიხი არაა)
         ).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(2 * c)))))
    m = _keep_long(m, axis=0, min_len=5 * c, max_thick=1.0 * c)
    if DEBUG_RULES:
        print('vfilter comps', cv2.connectedComponents(m)[0] - 1)
    if cv2.connectedComponents(m)[0] - 1 > VFILTER_MAX_COMPS:
        m = _drop_in_text(m, d, c)       # ცრუ ხაზების წყალობით ხაზების რაოდენობა ანომალიურია
    return cv2.resize(m, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)


def _drop_in_text(m, d, c):
    """ნამდვილი გამყოფი ხაზის ორივე მხარეს თეთრი ველია. ტექსტის შიგნით მიმავალი 'ხაზი'
    (ასოების შტრიხების თანხვედრა, განსაკუთრებით კირილიცაზე) ორივე მხარეს მელანს ხვდება."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    out = np.zeros_like(m)
    g1, g2 = max(2, int(0.4 * c)), max(4, int(1.6 * c))
    for i in range(1, n):
        x, y, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        l0, l1 = max(0, x - g2), max(0, x - g1)
        r0, r1 = min(W, x + w + g1), min(W, x + w + g2)
        if l1 <= l0 or r1 <= r0:
            out[lab == i] = 255
            continue
        dl = float((d[y:y + h, l0:l1].max(axis=1) > 0.5).mean())    # რამდენ სტრიქონს აქვს მელანი
        dr = float((d[y:y + h, r0:r1].max(axis=1) > 0.5).mean())
        if max(dl, dr) < SIDE_INK_MAX:
            out[lab == i] = 255
    return out


def _keep_long(mask, axis, min_len, max_thick):
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if axis == 0:
        length, thick = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_WIDTH]
    else:
        length, thick = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    area = st[:, cv2.CC_STAT_AREA]
    # გრძელი და, საშუალოდ, თხელი (მრუდე ხაზს bbox განიერი აქვს, მაგრამ ფართობი პატარა)
    keep = (length >= min_len) & ((area / np.maximum(length, 1)) <= max_thick)
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ---------------------------- ბლოკები ----------------------------

def find_blocks(ink, ch, vrules, hrules):
    """აბრუნებს: labels (ბლოკების ნომრები ყოველ პიქსელზე), boxes [(x0,y0,x1,y1,label)]."""
    seg = remove_small(ink, max(8, int(0.06 * ch * ch)))
    # დიდი მუქი ლაქები (მელნის ლაქა, დაზიანება) - შევსებული და დიდი
    n, labels, st, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
    bw, bh, ar = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    blot = (ar > 25 * ch * ch) & (ar > 0.55 * bw * bh)
    blot[0] = False
    seg[blot[labels]] = 0

    kx = max(3, int(WORD_JOIN * ch))
    ky = max(3, int(LINE_JOIN * ch))
    grown = cv2.dilate(seg, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))

    # ხაზების ბოლოებს ვაგრძელებთ: გამყოფი ხშირად ჰორიზონტალურ ხაზამდე ცოტა ადრე
    # მთავრდება და იმ "ხვრელიდან" მეზობელი სვეტები ერთმანეთს ეწებებოდა
    ext = max(5, int(2.5 * ch))
    barrier = (cv2.dilate(vrules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * ext + 1))) |
               cv2.dilate(hrules, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ext + 1, 3))))
    grown[barrier > 0] = 0

    n, labels, st, _ = cv2.connectedComponentsWithStats(grown, connectivity=4)

    # ბლოკის "ძირითადი" საზღვრები: ტექსტის პიქსელების 2-98%, რომ ვიწრო "კუდებმა"
    # (მაგ. ხაზის გასწვრივ გაპარული ნაწილი) კითხვის რიგი არ აურიოს
    s = 2
    lab_s, seg_s = labels[::s, ::s], seg[::s, ::s] > 0
    ys, xs = np.nonzero(seg_s & (lab_s > 0))
    ls = lab_s[ys, xs]
    order = np.argsort(ls, kind="stable")
    ls, ys, xs = ls[order], ys[order], xs[order]
    starts = np.searchsorted(ls, np.arange(n + 1))

    boxes, full = [], {}
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 1.5 * ch or h < 0.8 * ch:
            continue
        a_, b_ = starts[i], starts[i + 1]
        if b_ - a_ >= 20:
            rx0, rx1 = np.percentile(xs[a_:b_], [2, 98]) * s
            ry0, ry1 = np.percentile(ys[a_:b_], [2, 98]) * s
            box = (int(rx0), int(ry0), int(rx1) + s, int(ry1) + s, i)
        else:
            box = (x, y, x + w, y + h, i)
        boxes.append(box)
        full[i] = (x, y, x + w, y + h)
    return labels, boxes, full


def _clusters(intervals, shrink):
    """ინტერვალების გაერთიანება; აბრუნებს ჯგუფებს და მათ შორის დაშორებებს."""
    items = sorted(((a + shrink, max(a + shrink, b - shrink), idx) for a, b, idx in intervals))
    groups, gaps = [], []
    cur, cur_end = [items[0][2]], items[0][1]
    for a, b, idx in items[1:]:
        if a > cur_end:
            groups.append(cur)
            gaps.append((cur_end, a))
            cur, cur_end = [idx], b
        else:
            cur.append(idx)
            cur_end = max(cur_end, b)
    groups.append(cur)
    return groups, gaps


def order_boxes(boxes, ch, hrules):
    """კითხვის თანმიმდევრობა (XY-cut ბლოკების დონეზე)."""
    if len(boxes) <= 1:
        return list(boxes)
    by_id = {b[4]: b for b in boxes}

    # ჰორიზონტალური ხაზები (x0, x1, y-ის შუა)
    n, lab, st, cen = cv2.connectedComponentsWithStats(hrules, connectivity=8)
    hsegs = sorted((cen[i][1], st[i, 0], st[i, 0] + st[i, 2]) for i in range(1, n))
    # ერთი და იგივე ხაზი სვეტების გადაკვეთასთან ნაწილებადაა გაწყვეტილი - ვაერთებთ
    merged = [list(s) for s in hsegs]
    changed = True
    while changed:
        changed = False
        out = []
        for y, x0, x1 in merged:
            for m in out:
                if abs(m[0] - y) < 1.2 * ch and x0 < m[2] + 4 * ch and x1 > m[1] - 4 * ch:
                    m[0] = (m[0] + y) / 2
                    m[1], m[2] = min(m[1], x0), max(m[2], x1)
                    changed = True
                    break
            else:
                out.append([y, x0, x1])
        merged = out
    hsegs = [(x0, x1, y) for y, x0, x1 in merged]

    def recurse(ids):
        if len(ids) <= 1:
            return ids
        bs = [by_id[i] for i in ids]
        gx0, gx1 = min(b[0] for b in bs), max(b[2] for b in bs)

        # 0) ჰორიზონტალური ხაზი, რომელიც ჯგუფის თითქმის მთელ სიგანეზეა გაჭიმული და
        #    რომელსაც არც ერთი ბლოკი არ კვეთს - ზედა და ქვედა სტატიების საზღვარი
        cuts = []
        for x0, x1, y in hsegs:
            if min(x1, gx1) - max(x0, gx0) < 0.6 * (gx1 - gx0):
                continue
            if any(b[1] < y - 0.5 * ch and b[3] > y + 0.5 * ch for b in bs):
                continue
            if any((b[1] + b[3]) / 2 < y for b in bs) and any((b[1] + b[3]) / 2 > y for b in bs):
                cuts.append(y)
        if cuts:
            cuts = sorted(cuts)
            parts = [[] for _ in range(len(cuts) + 1)]
            for b in bs:
                parts[int(np.searchsorted(cuts, (b[1] + b[3]) / 2))].append(b[4])
            return [i for p in parts if p for i in recurse(p)]

        # 1) "ნამდვილი" ჰორიზონტალური გაყოფა: დიდი დაშორება ან მთელ სიგანეზე გაჭიმული ხაზი
        hg, hgaps = _clusters([(b[1], b[3], b[4]) for b in bs], 0)
        if len(hg) > 1:
            strong = []
            for (a, b_) in hgaps:
                big = (b_ - a) >= 2.5 * ch
                band = hrules[int(a):int(b_) + 1, gx0:gx1]
                ruled = band.size > 0 and (band.max(axis=0) > 0).mean() > 0.6
                strong.append(big or ruled)
            if any(strong):
                out, cur = [], list(hg[0])
                for g, s in zip(hg[1:], strong):
                    if s:
                        out += recurse(cur)
                        cur = list(g)
                    else:
                        cur += g
                return out + recurse(cur)

        # 2) სვეტებად (ოდნავ "შეკუმშული" ჩარჩოებით - მრუდე სვეტები ერთმანეთს რომ არ გადაფარავდეს)
        vg, _ = _clusters([(b[0], b[2], b[4]) for b in bs], 0.4 * ch)
        if len(vg) > 1:
            return [i for g in vg for i in recurse(g)]

        # 3) პატარა ჰორიზონტალური დაშორებებით (ერთ სვეტში აბზაცები)
        if len(hg) > 1:
            return [i for g in hg for i in recurse(g)]

        # 4) ვეღარ იყოფა - ზემოდან ქვემოთ
        return [b[4] for b in sorted(bs, key=lambda b: (b[1], b[0]))]

    return [by_id[i] for i in recurse([b[4] for b in boxes])]


# ---------------------------- სტატიებად გაერთიანება ----------------------------

def _leaf_stats(labels, clean_ink):
    """თითოეული ბლოკისთვის: ასოების (შეერთებული ნაწილების) მედიანური სიმაღლე და რაოდენობა."""
    H, W = labels.shape
    n, _, st, cen = cv2.connectedComponentsWithStats(clean_ink, connectivity=8)
    hs, ar = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    cx = np.clip(cen[1:, 0].astype(int), 0, W - 1)
    cy = np.clip(cen[1:, 1].astype(int), 0, H - 1)
    lab = labels[cy, cx]
    m = (ar >= 10) & (lab > 0)
    hs, lab = hs[m], lab[m]
    ref = float(np.median(hs)) if len(hs) else 20.0
    order = np.lexsort((hs, lab))
    hs, lab = hs[order], lab[order]
    uniq, start, cnt = np.unique(lab, return_index=True, return_counts=True)
    med = {int(u): float(hs[s + c // 2]) for u, s, c in zip(uniq, start, cnt)}
    num = {int(u): int(c) for u, c in zip(uniq, cnt)}
    return med, num, ref


def build_objects(boxes, full, labels, clean_ink, ch, keep_all=False):
    """ბლოკებიდან 'ობიექტებს' ვაკეთებთ: ყოველს აქვს ყუთი, ასოს სიმაღლე და სათაურის ნიშანი."""
    med, num, ref = _leaf_stats(labels, clean_ink)
    objs = []
    for b in boxes:
        lab = b[4]
        x0, y0, x1, y1 = [int(v) for v in b[:4]]
        w, h = x1 - x0, y1 - y0
        n_c = num.get(lab, 0)
        if not keep_all and (max(w, h) < 2.5 * ch or (n_c < 3 and max(w, h) < 6 * ch)):
            continue                                        # ლაქები და ნაკაწრები
        lh = med.get(lab, ref)
        head = bool(lh >= HEAD_RATIO * ref and not keep_all)
        if not keep_all and (lh < 0.5 * ref or (not head and n_c < 8)):
            continue                                        # ხაზების ნამტვრევები, მტვერი, ლაქები
        objs.append(dict(box=(x0, y0, x1, y1), labs=[lab], lh=lh, n=n_c, head=head))
    return objs, ref


def glue_headings(objs):
    """დიდი ასოებით დაწერილი სათაური ხშირად ასოებად/სიტყვებად იყოფა - ვაერთებთ."""
    idx = [i for i, o in enumerate(objs) if o["head"]]
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ai, i in enumerate(idx):
        ax0, ay0, ax1, ay1 = objs[i]["box"]
        for j in idx[ai + 1:]:
            bx0, by0, bx1, by1 = objs[j]["box"]
            hm = max(objs[i]["lh"], objs[j]["lh"])
            gx = max(bx0 - ax1, ax0 - bx1, 0)
            gy = max(by0 - ay1, ay0 - by1, 0)
            ovy = min(ay1, by1) - max(ay0, by0)
            ovx = min(ax1, bx1) - max(ax0, bx0)
            same_line = ovy >= 0.5 * min(ay1 - ay0, by1 - by0) and gx <= 2.5 * hm
            stacked = ovx >= 0.4 * min(ax1 - ax0, bx1 - bx0) and gy <= 1.5 * hm
            if same_line or stacked:
                parent[find(j)] = find(i)
    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    merged = []
    for g in groups.values():
        bs = [objs[i]["box"] for i in g]
        merged.append(dict(box=(min(b[0] for b in bs), min(b[1] for b in bs),
                                max(b[2] for b in bs), max(b[3] for b in bs)),
                           labs=[l for i in g for l in objs[i]["labs"]],
                           lh=max(objs[i]["lh"] for i in g), n=sum(objs[i]["n"] for i in g), head=True))
    return [o for o in objs if not o["head"]] + merged


def closed_cells(R, ch):
    """ხაზებით მთლიანად შემოსაზღვრული არეები (ჩარჩოში ჩასმული განცხადებები, უჯრები)."""
    H, W = R.shape
    k = max(3, int(0.5 * ch)) | 1
    free = (cv2.dilate(R, np.ones((k, k), np.uint8)) == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H:
            continue                                       # გვერდის კიდეს ეხება - ე.ი. გარეთაა
        if w * h > 0.30 * W * H or w < 4 * ch or h < 3 * ch:
            continue
        keep[i] = True
    return np.where(keep[lab], lab, 0).astype(np.int32)


def _band_has_line(Rb, x0, x1, y0, y1, frac=0.4):
    H, W = Rb.shape
    x0, x1, y0, y1 = max(0, int(x0)), min(W, int(x1)), max(0, int(y0)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(Rb[y0:y1, x0:x1].any(axis=0).mean() >= frac)


def _rule_extent(Rh, x_lo, x_hi, y0, y1, ch):
    """ხაზის სიგრძე (პიქსელებში), რომელიც [y0,y1] ზოლში [x_lo,x_hi] შუალედს კვეთს."""
    H, W = Rh.shape
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x_lo, x_hi = max(0, int(x_lo)), min(W, int(x_hi))
    if y1 <= y0 or x_hi <= x_lo:
        return 0
    cols = Rh[y0:y1].any(axis=0).astype(np.uint8)
    cols = cv2.dilate(cols.reshape(1, -1), np.ones((1, max(3, int(2 * ch))), np.uint8)).ravel() > 0
    if not cols[x_lo:x_hi].any():
        return 0
    d = np.diff(np.concatenate(([0], cols.astype(np.int8), [0])))
    best = 0
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if b > x_lo and a < x_hi:
            best = max(best, b - a)
    return best


def _union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def make_units(objs, ch, hrules, frames, cell, shape):
    """ობიექტებს სტატიებად აჯგუფებს. აბრუნებს units = [{'members': [ობიექტის ნომრები], 'box': ...}]."""
    H, W = shape
    n = len(objs)
    if n == 0:
        return []
    Rh = ((hrules | frames) > 0)
    B = np.array([o["box"] for o in objs], dtype=np.float64)

    for o in objs:                                          # რომელ "უჯრაშია" ობიექტი
        x0, y0, x1, y1 = o["box"]
        pts = [((x0 + x1) // 2, (y0 + y1) // 2), (x0 + (x1 - x0) // 4, y0 + (y1 - y0) // 4),
               (x1 - (x1 - x0) // 4, y0 + (y1 - y0) // 4), (x0 + (x1 - x0) // 4, y1 - (y1 - y0) // 4),
               (x1 - (x1 - x0) // 4, y1 - (y1 - y0) // 4)]
        vals = [int(cell[min(H - 1, py), min(W - 1, px)]) for px, py in pts]
        o["cell"] = max(set(vals), key=vals.count)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, o in enumerate(objs):                            # ერთი ჩარჩოს შიგნით ყველაფერი ერთია
        if o["cell"]:
            if o["cell"] in first:
                union(i, first[o["cell"]])
            else:
                first[o["cell"]] = i

    order = sorted(range(n), key=lambda i: (B[i, 1], B[i, 0]))
    for i in order:                                         # სვეტში ზემოდან ქვემოთ "ჯაჭვი"
        if objs[i]["head"]:
            continue
        x0, y0, x1, y1 = B[i]
        w = x1 - x0
        ov = np.minimum(x1, B[:, 2]) - np.maximum(x0, B[:, 0])
        is_head = np.array([o["head"] for o in objs])
        above = (B[:, 3] <= y0 + np.where(is_head, 1.5, 0.5) * ch) & \
                (ov >= 0.5 * np.minimum(w, B[:, 2] - B[:, 0])) & ((B[:, 1] + B[:, 3]) < (y0 + y1))
        above[i] = False
        if not above.any():
            continue
        cand = np.where(above)[0]
        j = int(cand[np.argmax(B[cand, 3])])                # უახლოესი ზემოთ
        gap = max(0.0, y0 - B[j, 3])
        if objs[i]["cell"] != objs[j]["cell"]:
            continue
        if objs[j]["head"]:
            # სათაურის ქვეშ მოკლე ხაზი (ხაზგასმა) ჩვეულებრივია, მაგრამ მთელ გვერდზე გაჭიმული
            # ხაზი (მაგ. გვერდის თავსართის ქვეშ) სტატიების საზღვარია
            ext = _rule_extent(Rh, max(x0, B[j, 0]), min(x1, B[j, 2]), B[j, 3] - 0.3 * ch, y0 + 0.3 * ch, ch)
            ok = gap <= 6 * ch and ext <= 1.8 * max(w, B[j, 2] - B[j, 0]) + 4 * ch
        else:
            ok = gap <= TEXT_GAP * ch and not _band_has_line(
                Rh, max(x0, B[j, 0]), min(x1, B[j, 2]), B[j, 3] - 0.3 * ch, y0 + 0.3 * ch)
        if ok:
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    units = [dict(members=m, box=_union_box([tuple(B[i]) for i in m]),
                  size=sum(objs[i]["n"] for i in m), cell=max(objs[i]["cell"] for i in m))
             for m in groups.values()]
    if MAX_BLOCKS and MAX_BLOCKS > 0:
        units = cap_units(units, objs, B, Rh, ch, MAX_BLOCKS)
    return units


def cap_units(units, objs, B, Rh, ch, maxn):
    """1) ერთმანეთში მოქცეულ ნაწილებს აერთიანებს; 2) სანამ ნაწილები maxn-ზე მეტია,
    ყველაზე პატარას აერთიანებს ყველაზე შესაფერის მეზობელთან."""
    cx, cy = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2
    owner = np.zeros(len(objs), dtype=np.int64)
    alive = {}
    for k, u in enumerate(units):
        alive[k] = u
        for i in u["members"]:
            owner[i] = k
    nxt = len(units)

    def merge(k, m):
        nonlocal nxt
        u, v = alive.pop(k), alive.pop(m)
        mem = u["members"] + v["members"]
        alive[nxt] = dict(members=mem, box=_union_box([tuple(B[i]) for i in mem]),
                          size=u["size"] + v["size"], cell=max(u["cell"], v["cell"]))
        for i in mem:
            owner[i] = nxt
        nxt += 1

    def contained():
        """ნაწილი, რომლის ცენტრი სხვა ნაწილის რომელიმე ბლოკშია (არა გარსმომვლებ ყუთში - L-ფორმა
        ყუთის შიგნით სხვა სტატიებს მოიცავს)."""
        out = {}
        for k, u in alive.items():
            big = max(u["members"], key=lambda i: objs[i]["n"])
            px, py = cx[big], cy[big]
            hit = (B[:, 0] < px) & (B[:, 2] > px) & (B[:, 1] < py) & (B[:, 3] > py) & (owner != k)
            if hit.any():
                out[k] = int(owner[np.where(hit)[0][0]])
        return out

    while True:
        pairs = contained()
        if pairs:
            k = min(pairs, key=lambda q: alive[q]["size"])
            merge(k, pairs[k])
            continue
        if len(alive) <= maxn:
            break
        ids = list(alive)
        ctr = np.array([[(alive[q]["box"][0] + alive[q]["box"][2]) / 2,
                         (alive[q]["box"][1] + alive[q]["box"][3]) / 2] for q in ids])
        d2 = ((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        near = np.argsort(d2, axis=1)[:, :8]

        def pair_cost(k, m):
            u, v = alive[k], alive[m]
            ux0, uy0, ux1, uy1 = u["box"]
            vx0, vy0, vx1, vy1 = v["box"]
            gx, gy = max(vx0 - ux1, ux0 - vx1, 0), max(vy0 - uy1, uy0 - vy1, 0)
            ovx = min(ux1, vx1) - max(ux0, vx0)
            ovy = min(uy1, vy1) - max(uy0, vy0)
            stacked = ovx >= 0.3 * min(ux1 - ux0, vx1 - vx0)
            side = ovy >= 0.3 * min(uy1 - uy0, vy1 - vy0)
            d = gy if stacked else (gx if side else gx + gy + 20 * ch)
            bx0, by0, bx1, by1 = min(ux0, vx0), min(uy0, vy0), max(ux1, vx1), max(uy1, vy1)
            excess = (bx1 - bx0) * (by1 - by0) - (ux1 - ux0) * (uy1 - uy0) - (vx1 - vx0) * (vy1 - vy0)
            inside = (cx > bx0) & (cx < bx1) & (cy > by0) & (cy < by1) & (owner != k) & (owner != m)
            cost = d / ch + max(excess, 0) / (50.0 * ch * ch) + 10.0 * int(inside.sum())
            if stacked and gy > 0 and _band_has_line(Rh, max(ux0, vx0), min(ux1, vx1),
                                                     min(uy1, vy1) - 0.3 * ch, max(uy0, vy0) + 0.3 * ch):
                cost += 15.0
            if u["cell"] != v["cell"]:
                cost += 20.0
            sm = min(u["size"], v["size"])
            return cost + 3.0 * sm / (sm + 300.0)           # პატარა ნაწილი უფრო იაფად ერწყმის

        best, best_cost = None, 1e18
        for a_i, k in enumerate(ids):
            for b_i in near[a_i]:
                m = ids[int(b_i)]
                c = pair_cost(k, m)
                if c < best_cost:
                    best, best_cost, best_k = m, c, k
        k = best_k
        merge(k, best)
    return list(alive.values())


# ---------------------------- OCR ----------------------------

def ocr_image(img, psm):
    """აბრუნებს (ტექსტი, საშუალო სანდოობა 0-100)."""
    img = cv2.copyMakeBorder(img, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    cfg = f"--oem 1 --psm {psm}"
    d = pytesseract.image_to_data(img, lang=CUR_LANG, config=cfg, output_type=pytesseract.Output.DICT)
    lines, confs, cur, cur_key = [], [], [], None
    for txt, conf, b, p, l in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
        key = (b, p, l)
        if key != cur_key:
            if cur:
                lines.append(" ".join(cur))
            cur, cur_key = [], key
        if txt.strip():
            cur.append(txt)
            if float(conf) >= 0:
                confs.append(float(conf))
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines), (float(np.mean(confs)) if confs else 0.0)


def ocr_object(sources, labels, obj, full, ch):
    """sources: ერთი ან ორი სურათი (ნაცრისფერი / შავ-თეთრი). თუ ორია, ორივეს კითხულობს
    და იღებს იმას, რომელშიც Tesseract უფრო დარწმუნებულია."""
    labs = obj["labs"]
    x0 = min(full[l][0] for l in labs)
    y0 = min(full[l][1] for l in labs)
    x1 = max(full[l][2] for l in labs)
    y1 = max(full[l][3] for l in labs)
    p = int(0.3 * ch)
    h, w = labels.shape
    xa, ya, xb, yb = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
    sub = labels[ya:yb, xa:xb]
    mask = (sub == labs[0]) if len(labs) == 1 else np.isin(sub, labs)
    best = ("", -1.0)
    for src in sources:
        crop = np.where(mask, src[ya:yb, xa:xb], 255).astype(np.uint8)   # მეზობელი ბლოკები გარეთ
        res = ocr_image(crop, 6)
        if res[1] > best[1]:
            best = res
    return best


def join_parts(parts):
    """ბლოკის ნაწილებს (სათაური, სვეტები) აერთებს; სვეტის ბოლოს გადატანილ სიტყვას აგრძელებს."""
    out = parts[0]
    for q in parts[1:]:
        if JOIN_HYPHENS and re.search(rf"[{GEO}][-–¬=]$", out) and re.match(rf"[{GEO}]", q):
            out = out[:-1] + q
        else:
            out += "\n" + q
    return out


_BAD = re.compile(r"[^\u10D0-\u10FF\u0400-\u04FFA-Za-z0-9\s.,;:!?()\[\]\"'«»„“”‘’\-–—%/№§*+=<>&]")


CUR_LANG = LANG        # თითო PDF-ზე process_pdf ცვლის


def clean_text(t):
    t = t.replace("\f", "").strip()
    if not t:
        return ""
    if CLEAN_SYMBOLS:
        t = "".join(chr(ord(c) - 0x1C90 + 0x10D0) if 0x1C90 <= ord(c) <= 0x1CBA else c for c in t)
        t = re.sub(r"[|‖¦]", " ", t)
        t = _BAD.sub("", t)
    if JOIN_HYPHENS:
        t = re.sub(rf"([{GEO}])[-–—¬=]\s*\n\s*([{GEO}])", r"\1\2", t)
        t = re.sub(r"([\u0400-\u04FF])[-¬]\s*\n\s*([\u0430-\u04FF])", r"\1\2", t)
    if MERGE_LINES:
        t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def looks_like_text(t):
    n = len(re.findall(f"[{GEO}\u0400-\u04FF]", t))       # ქართული ან კირილიცა
    return n >= 3 and n >= 0.4 * len(re.sub(r"\s", "", t))


# ---------------------------- debug ----------------------------

_PALETTE = [(0, 0, 230), (230, 0, 0), (0, 150, 0), (0, 130, 255), (200, 0, 200),
            (0, 190, 190), (120, 60, 0), (255, 110, 0)]


def save_debug(gray, units, objs, vrules, hrules, path):
    """თითოეული ნაწილი თავისი ფერით: სქელი ჩარჩო = ნაწილი, ნომერი = კითხვის რიგი,
    ფერადი ფონი = რა ბლოკებისგან შედგება, მწვანე = ნაპოვნი გამყოფი ხაზები."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    th = max(2, gray.shape[1] // 700)
    over = vis.copy()
    for k, u in enumerate(units):
        c = _PALETTE[k % len(_PALETTE)]
        for i in u["members"]:
            x0, y0, x1, y1 = objs[i]["box"]
            cv2.rectangle(over, (x0, y0), (x1, y1), c, -1)
    vis = cv2.addWeighted(over, 0.16, vis, 0.84, 0)
    vis[(vrules | hrules) > 0] = (60, 200, 60)
    for k, u in enumerate(units):
        c = _PALETTE[k % len(_PALETTE)]
        x0, y0, x1, y1 = [int(v) for v in u["box"]]
        area = sum((objs[i]["box"][2] - objs[i]["box"][0]) * (objs[i]["box"][3] - objs[i]["box"][1])
                   for i in u["members"])
        if area >= 0.8 * (x1 - x0) * (y1 - y0):
            cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2 * th)           # კომპაქტური ნაწილი
        else:                                                           # L-ფორმა: თითოეული ბლოკი ცალკე
            for i in u["members"]:
                bx0, by0, bx1, by1 = objs[i]["box"]
                cv2.rectangle(vis, (bx0, by0), (bx1, by1), c, 2 * th)
            x0, y0 = objs[u["members"][0]]["box"][:2]
        lab = str(k + 1)
        sc = 0.9 * th
        (tw, tg), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        cv2.rectangle(vis, (x0, y0), (x0 + tw + 10, y0 + tg + 14), c, -1)
        cv2.putText(vis, lab, (x0 + 5, y0 + tg + 5), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th)
    s = 1800.0 / max(vis.shape)
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------- PDF ----------------------------

def layout_units(clean_ink, ch, vrules, hrules, frames):
    """აბრუნებს (objs, units): კითხვის თანმიმდევრობით დალაგებულ ნაწილებს."""
    labels, blocks, full = find_blocks(clean_ink, ch, vrules, hrules)
    keep_all = not (MAX_BLOCKS and MAX_BLOCKS > 0)
    objs, ref = build_objects(blocks, full, labels, clean_ink, ch, keep_all)
    if keep_all:
        units = [dict(members=[i], box=o["box"]) for i, o in enumerate(objs)]
    else:
        objs = glue_headings(objs)
        R = ((vrules | hrules | frames) > 0).astype(np.uint8) * 255
        cell = closed_cells(R, ch)
        units = make_units(objs, ch, hrules, frames, cell, clean_ink.shape)
    # კითხვის რიგი: ჯერ ყველა ობიექტი ერთად (XY-cut), ნაწილის რიგი = მისი პირველი ობიექტის რიგი
    ordered = order_boxes([(*o["box"], i) for i, o in enumerate(objs)], ch, hrules)
    rank = {b[4]: r for r, b in enumerate(ordered)}
    for u in units:
        u["members"].sort(key=lambda i: rank[i])
    units.sort(key=lambda u: rank[u["members"][0]])
    return labels, full, objs, units


def process_page(doc, pno, base, debug_dir):
    gray = load_page(doc, doc[pno])
    norm, ink, ch = prepare_image(gray)
    vrules, hrules, frames = find_rules(ink, ch)
    rules = cv2.dilate(vrules | hrules, np.ones((5, 5), np.uint8))
    clean_ink = cv2.bitwise_and(ink, cv2.bitwise_not(rules))
    binary = cv2.bitwise_not(clean_ink)          # შავ-თეთრი, ხაზების გარეშე
    gray = norm.copy()
    gray[rules > 0] = 255                        # ნაცრისფერი, ხაზების გარეშე
    sources = [gray, binary] if QUALITY == "best" else [gray]

    texts, units = [], []
    if LAYOUT == "tesseract":
        texts = [clean_text(ocr_image(gray, 3)[0])]
    else:
        labels, full, objs, units = layout_units(clean_ink, ch, vrules, hrules, frames)
        if DEBUG_IMAGES:
            save_debug(norm, units, objs, vrules, hrules,
                       os.path.join(debug_dir, f"{base}_p{pno + 1:03d}.jpg"))
        for u in units:
            parts = []
            for i in u["members"]:
                txt, conf = ocr_object(sources, labels, objs[i], full, ch)
                if conf >= MIN_CONFIDENCE:                # ლაქები, ორნამენტები, სურათები
                    t = clean_text(txt)
                    if looks_like_text(t):
                        parts.append(t)
            if parts:
                texts.append(join_parts(parts))
    return texts, len(units), ch


def lang_for(pdf_path):
    sub = os.path.basename(os.path.dirname(os.path.abspath(pdf_path))).lower()
    in_root = os.path.abspath(os.path.dirname(pdf_path)) == os.path.abspath(INPUT_PDF_FOLDER)
    return LANG if in_root else LANG_BY_FOLDER.get(sub, LANG)


def process_pdf(pdf_path, output_folder):
    global CUR_LANG
    CUR_LANG = lang_for(pdf_path)
    name = os.path.basename(pdf_path)
    base = os.path.splitext(name)[0]
    out_path = os.path.join(output_folder, base + ".txt")
    debug_dir = os.path.join(output_folder, "_debug")

    t0 = time.time()
    doc = fitz.open(pdf_path)
    parts = []
    for pno in range(len(doc)):
        texts, nb, ch = process_page(doc, pno, base, debug_dir)
        parts.append(f"===== გვერდი {pno + 1} =====\n\n" + "\n\n".join(texts) + "\n\n")
        print(f"   {name}: გვერდი {pno + 1}/{len(doc)} ({nb} ბლოკი)", flush=True)
    doc.close()

    tmp = out_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(parts)
    os.replace(tmp, out_path)
    return name, time.time() - t0


def check_languages():
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception as e:
        sys.exit(f"Tesseract ვერ გაეშვა ({e}). შეამოწმეთ TESSERACT_CMD გზა.")
    need = set(LANG.split("+"))
    if os.path.isdir(INPUT_PDF_FOLDER):
        for d in os.listdir(INPUT_PDF_FOLDER):
            if os.path.isdir(os.path.join(INPUT_PDF_FOLDER, d)):
                need |= set(LANG_BY_FOLDER.get(d.lower(), LANG).split("+"))
    missing = sorted(l for l in need if l not in langs)
    if missing:
        sys.exit(f"Tesseract-ში არ არის ენა/ენები: {', '.join(missing)}.\n"
                 f"ჩამოტვირთეთ <ენა>.traineddata და ჩააგდეთ tessdata საქაღალდეში.\n"
                 f"ახლა ხელმისაწვდომია: {', '.join(sorted(langs))}")


def main():
    check_languages()
    os.makedirs(OUTPUT_TXT_FOLDER, exist_ok=True)
    if DEBUG_IMAGES:
        os.makedirs(os.path.join(OUTPUT_TXT_FOLDER, "_debug"), exist_ok=True)

    pdfs = []
    for root, _d, files in os.walk(INPUT_PDF_FOLDER):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    pdfs.sort()
    if not pdfs:
        print(f"'{INPUT_PDF_FOLDER}' საქაღალდეში PDF ფაილები ვერ მოიძებნა.")
        return

    todo = []
    for p in pdfs:
        txt = os.path.join(OUTPUT_TXT_FOLDER, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if SKIP_EXISTING and os.path.exists(txt):
            print(f"გამოტოვებულია (უკვე არსებობს): {os.path.basename(p)}")
            continue
        todo.append(p)

    print(f"\nსულ {len(pdfs)} PDF, დასამუშავებელი {len(todo)}. პარალელურად: {WORKERS}\n")
    errors = []
    if WORKERS == 1:
        for p in todo:
            try:
                n, sec = process_pdf(p, OUTPUT_TXT_FOLDER)
                print(f"-> მზადაა: {n} ({sec:.0f} წმ)")
            except Exception as e:
                errors.append((p, e))
                print(f"-> შეცდომა: {p}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(process_pdf, p, OUTPUT_TXT_FOLDER): p for p in todo}
            for fut in as_completed(futs):
                try:
                    n, sec = fut.result()
                    print(f"-> მზადაა: {n} ({sec:.0f} წმ)")
                except Exception as e:
                    errors.append((futs[fut], e))
                    print(f"-> შეცდომა: {futs[fut]}: {e}")

    print(f"\nდასრულდა. შედეგები: {OUTPUT_TXT_FOLDER}")
    if errors:
        print(f"შეცდომით დასრულდა {len(errors)} ფაილი:")
        for p, e in errors:
            print(f"   {p}: {e}")


if __name__ == "__main__":
    main()
