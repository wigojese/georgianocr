
from pathlib import Path
import argparse
import re
import sys
import traceback

import cv2
import numpy as np
import pymupdf
import pytesseract
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# PDF-ების საწყისი საქაღალდე
INPUT_DIR = Path("input_pdfs")

# ყველა ტექსტური ფაილი აქ შეინახება
OUTPUT_DIR = Path("output")

# Tesseract-ის მდებარეობა Windows-ზე
TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Tesseract-ის მოდელების საქაღალდე
TESSDATA_DIR = Path(
    r"C:\Program Files\Tesseract-OCR\tessdata"
)

# ორივე მოდელი უნდა არსებობდეს tessdata საქაღალდეში
OCR_LANG = "kat"

# 250 DPI უფრო სწრაფია; წვრილი ტექსტისთვის სცადე 300
DPI = 250

# Tesseract-ის გვერდის განლაგების რეჟიმი:
# 3 = ავტომატური განლაგება, რამდენიმე სვეტის ჩათვლით
PSM = 3

# მცირე დახრის გასწორება
DESKEW = True

# თუ Trueა, უკვე არსებული TXT ფაილები თავიდან დამუშავდება
OVERWRITE = False


# ============================================================
# INITIALIZATION
# ============================================================

pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)


def validate_setup():
    if not TESSERACT_EXE.exists():
        raise FileNotFoundError(
            f"Tesseract ვერ მოიძებნა:\n{TESSERACT_EXE}\n"
            "შეცვალე TESSERACT_EXE ცვლადში მისი რეალური მისამართი."
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"PDF-ების საქაღალდე ვერ მოიძებნა: {INPUT_DIR}"
        )

    for lang in OCR_LANG.split("+"):
        model = TESSDATA_DIR / f"{lang}.traineddata"
        if not model.exists():
            raise FileNotFoundError(
                f"ენის მოდელი ვერ მოიძებნა:\n{model}\n"
                f"შეამოწმე OCR_LANG და TESSDATA_DIR."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def render_page(page):
    """PDF გვერდის გამოსახულებად გადაყვანა."""
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(DPI / 72, DPI / 72),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )

    image = np.frombuffer(
        pix.samples, dtype=np.uint8
    ).reshape(pix.height, pix.width)

    return image


def deskew_image(image):
    """გვერდის მცირე დახრის გასწორება."""
    inverted = cv2.bitwise_not(image)

    _, binary = cv2.threshold(
        inverted,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 100:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # დიდი კუთხის კორექცია ხშირად საზიანოა
    if abs(angle) > 5 or abs(angle) < 0.15:
        return image

    height, width = image.shape

    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess(image):
    """სწრაფი, მსუბუქი დამუშავება."""
    if DESKEW:
        image = deskew_image(image)

    return image


# ============================================================
# OCR
# ============================================================



def run_ocr(image):
    """ერთი OCR გაშვება ერთ გვერდზე."""
    import os

    config = f'--oem 1 --psm {PSM} -c preserve_interword_spaces=1'

    pil_image = Image.fromarray(image)

    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

    return pytesseract.image_to_string(
        pil_image,
        lang=OCR_LANG,
        config=config,
        timeout=300,
    )




def clean_text(text):
    """ტექსტის მსუბუქი გასუფთავება."""
    text = text.replace("\x0c", "")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(pdf_path):
    """
    ერთი PDF -> ერთი TXT ფაილი.
    TXT-ის სახელი ემთხვევა PDF-ის სახელს.
    """

    output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

    if output_path.exists() and not OVERWRITE:
        if output_path.stat().st_size > 0:
            print(f"[SKIP] {pdf_path.name}: TXT უკვე არსებობს")
            return

    print(f"\n[PDF] {pdf_path.name}", flush=True)

    document = pymupdf.open(pdf_path)

    # დროებითი ფაილი იცავს უკვე დაწერილ ტექსტს პროცესის
    # შეწყვეტისას და საბოლოო ფაილს მხოლოდ ბოლოს ანაცვლებს.
    temp_path = output_path.with_suffix(".tmp")

    try:
        with temp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as out:

            for index, page in enumerate(document):
                page_number = index + 1

                try:
                    image = render_page(page)
                    image = preprocess(image)
                    text = clean_text(run_ocr(image))

                    out.write(
                        f"\n\n===== გვერდი {page_number} =====\n\n"
                    )
                    out.write(text)
                    out.write("\n")
                    out.flush()

                    print(
                        f"  გვერდი {page_number}/{len(document)} OK",
                        flush=True,
                    )

                except Exception as error:
                    print(
                        f"  [ERROR] გვერდი {page_number}: {error}",
                        flush=True,
                    )

                    # შეცდომის აღრიცხვა TXT-ში, რათა გვერდი
                    # შეუმჩნევლად არ დაიკარგოს.
                    out.write(
                        f"\n\n===== გვერდი {page_number} =====\n"
                        f"[OCR ERROR: {error}]\n"
                    )
                    out.flush()

        temp_path.replace(output_path)

        print(f"[DONE] {output_path}", flush=True)

    except Exception:
        print(
            f"[FATAL] PDF ვერ დამუშავდა: {pdf_path}",
            flush=True,
        )
        traceback.print_exc()

        # დროებითი ფაილი რჩება აღდგენისთვის
        raise

    finally:
        document.close()


# ============================================================
# MAIN
# ============================================================

def main():

    global INPUT_DIR, OUTPUT_DIR, OCR_LANG, DPI, OVERWRITE


    parser = argparse.ArgumentParser(
        description="ძველი ქართული გაზეთების OCR"
    )

    parser.add_argument(
        "--input",
        default=str(INPUT_DIR),
        help="PDF-ების საქაღალდე",
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="TXT ფაილების საქაღალდე",
    )

    parser.add_argument(
        "--lang",
        default=OCR_LANG,
        help="Tesseract-ის ენა, მაგალითად kat ",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="სკანის რეზოლუცია, მაგალითად 200, 250 ან 300",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="არსებული TXT ფაილების ხელახლა შექმნა",
    )

    args = parser.parse_args()


    INPUT_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    OCR_LANG = args.lang
    DPI = args.dpi
    OVERWRITE = OVERWRITE or args.overwrite

    try:
        validate_setup()
    except Exception as error:
        print(f"[SETUP ERROR] {error}")
        sys.exit(1)

    pdf_files = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"PDF ფაილები ვერ მოიძებნა: {INPUT_DIR}")
        return

    print(f"ნაპოვნია {len(pdf_files)} PDF ფაილი.")
    print(f"ენა: {OCR_LANG}")
    print(f"DPI: {DPI}")
    print(f"შედეგები: {OUTPUT_DIR.resolve()}")

    completed = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path)
            completed += 1
        except Exception as error:
            failed += 1
            print(
                f"[FAILED] {pdf_path.name}: {error}",
                flush=True,
            )

    print("\n================================")
    print("დამუშავება დასრულდა.")
    print(f"დამუშავებული/გამოტოვებული: {completed}")
    print(f"შეცდომით დასრულებული: {failed}")
    print(f"შედეგების საქაღალდე: {OUTPUT_DIR.resolve()}")
    print("================================")


if __name__ == "__main__":
    main()
