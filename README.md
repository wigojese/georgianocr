# Georgian historical newspaper OCR

OCR for scanned Georgian newspapers and journals from the 1910s–1920s (PDF → text), built on Tesseract.

`georgian_ocr.py` is version 6. Versions 2–5 and the simple one-pass script are kept in `legacy/` for comparison.

## Results

The test pages are *Guriis Erobis Moambe* 1920 №1 p.1 (clean) and №6 p.2 (dark, low contrast). Each was scored against a hand-typed transcription in `ground_truth/` using `evaluate.py`.

|                         | CER (characters) | WER (words) |
|-------------------------|-----------------:|------------:|
| legacy simple (psm 3)   | 7.4 %            | 42.7 %      |
| legacy v2               | 46.6 %           | 74.1 %      |
| legacy v4               | 48.5 %           | 74.6 %      |
| legacy v5               | 35.6 %           | 68.5 %      |
| **v6**                  | **1.2 %**        | **9.1 %**   |

The ground truth is still small (about 2,000 characters). Adding more pages makes these numbers more reliable (see below).

## What v6 does differently, and why

1. **Column layout instead of blob merging.** v2–v5 built blocks by smearing ink and gluing pieces. In justified text the gaps between words are as wide as the gaps between columns, so a column got cut into vertical strips and lines were read out of order. That is where most of their 35–60 % error came from. v6 uses a recursive XY-cut:
   - It cuts only at real rules (long, thin lines) and at white gutters that run the full height of a region.
   - Where a gutter is blocked by something spanning several columns (the masthead, a wide heading), it cuts horizontally just above and below that element.
   - A typical page comes out as header + 2 columns. Each column is read by Tesseract as a whole, so headings stay attached to their text.
2. **Native resolution.** On these scans upscaling made recognition worse. Pages are resized only if the line pitch is outside 18–48 px.
3. **Three image variants per line.** Each block is read three ways: normalised grey, Sauvola binary, and grey upscaled ×1.5. For each line the best reading is kept, scored by confidence plus the share of words found in the lexicon.
4. **Fake punctuation removed.** Tesseract invents line-end periods and turns paper specks into `.` or `„`. Every trailing `. , : ;`, every quote, and every leading quote is checked against the pixels. A real period is a small blob at the baseline; a real quote is two marks or one large one. A hyphen-shaped mark that was read as `.` or `“` is turned back into `-`.
5. **Words broken across lines.** All line geometry is kept until the end. A line-end fragment is joined with the start of the next line when:
   - there is a hyphen of any kind (`- – — ¬ = ~`, or a hyphen Tesseract printed as a separate token);
   - or the hyphen was lost or misread, the line reaches the right margin, and the joined word is in the lexicon.

   When both halves are real words on their own (`და` + `სხვა`), the pair is joined only if the joined form is frequent in your texts. Joins also work across column and page boundaries.

   The lexicon is `georgian_words.txt`: 197k word forms taken from Tesseract's `kat` model. Words that occur at least 3 times in your existing outputs are added to it, which picks up 1920s spellings such as `ერთიანათ` and `იმდენათ`.
6. **Low-confidence text is kept.** Only decorative masthead lettering and isolated junk lines (stamps, ornaments) are dropped. The running page header is removed by default.
7. **Paragraphs and headings.** Paragraphs are detected from indentation and short last lines. Headings are detected by larger letter height or centring, and appear as their own paragraph (`--mark-headings` prefixes them with `## `).

## Installation

```
pip install -r requirements.txt
```

- **Tesseract 5** with Georgian. On Windows use the UB Mannheim installer and set `TESSERACT_CMD` at the top of the script.
- **Strongly recommended:** download `kat.traineddata` from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best). Put it in its own folder, e.g. `C:\tessdata_best\`, and set `TESSDATA_DIR` to that folder (or pass `--tessdata`). On the test pages the best model lowered CER from 2.2 % to 1.65 %.

## Usage

```
python georgian_ocr.py                                   # every PDF in pdf_files/ -> extracted_texts/
python georgian_ocr.py some.pdf --pages 1-3 --debug      # selected pages + layout images
python georgian_ocr.py pdf_files --overwrite --fast      # re-run everything, one variant (≈3× faster)
```

Useful options:

| option | effect |
|---|---|
| `--debug` | saves `extracted_texts/_debug/*.jpg` showing blocks (red, numbered in reading order), lines (blue), rules (green) and page headers (grey) |
| `--fast` | one image variant instead of three |
| `--keep-headers` | keep the running header line (newspaper name, page number) |
| `--mark-headings` | prefix headings with `## ` |
| `--lines` | keep the original line breaks (no paragraph joining or de-hyphenation) |
| `--lang kat+rus` | Georgian text with Russian inserts |
| `--workers N` | number of pages processed in parallel |

Speed: about 12 s per page per CPU core in the default (best) mode. 68 pages took 3.5 minutes on 4 cores.

## Measuring quality: `evaluate.py`

1. Type up 2–3 paragraphs from a page, exactly as printed, and save them as `ground_truth/<pdf name>_p<page>.txt`, e.g. `Guriis_Erobis_Moambe_1920_N1_p001.txt`. Put each passage on its own line; passages can come from anywhere on the page. Include hard pages: bold type, dark scans, small print.
2. Run:

```
python evaluate.py extracted_texts                    # score one output folder
python evaluate.py extracted_texts other_folder       # compare two versions side by side
```

Use this before and after every change. Without it, tuning is guesswork, and each fix can quietly break another page.

## Known limits / next steps

- **Bold and decorative type** is the weakest point. The bold quotation on N1 p.1 and the bold decisions page N6 p.5 have most of the remaining errors. The stock `kat` model was trained on modern fonts.
- **Fine-tuning `kat` on these newspapers** is the next big step. Use [tesstrain](https://github.com/tesseract-ocr/tesstrain) with a few hundred corrected text lines (line image + transcription). `--debug` and the line boxes give you the line crops, and correcting v6 output is much faster than typing from scratch.
- **Mastheads and date bars** in decorative fonts come out garbled or are dropped.
- **4-column newspapers** (e.g. *Sakartvelos Respublika*): only the debug overlays of those pages were available for testing, not the PDFs. Run a few pages with `--debug` and check the block layout.
