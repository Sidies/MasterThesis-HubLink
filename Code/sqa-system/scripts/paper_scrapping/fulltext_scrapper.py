import os
import json
import re
import unicodedata
from tqdm import tqdm
import pymupdf


def extract_texts_from_folder(pdf_folder):
    """
    Extracts text from all PDF files in the specified folder.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
    """
    pdf_texts = {}
    file_names = os.listdir(pdf_folder)
    file_names = [f for f in file_names if f.endswith('.pdf')]
    for filename in tqdm(file_names, desc="Processing PDFs"):
        pdf_path = os.path.join(pdf_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        if text:
            pdf_texts[filename] = text

    return pdf_texts


def clean_text(text):
    """
    After pymupdf extraction, we had several issues with the text that needed to be cleaned.
    This function cleans the text by removing unwanted characters.

    Args:
        text (str): The text to be cleaned.
    """
    text = re.sub(r'\n', ' ', text)

    footer_pattern = re.compile(
        r'\d+\s*'  # Match one or more digits followed by optional whitespace
        # Non-greedy match for any characters except uppercase letters (optional additional text)
        r'(?:[^A-Z]*?)'
        r'Authorized licensed use limited to: KIT Library\.\s*'  # Match the fixed phrase
        # Match date and time
        r'Downloaded on [A-Za-z]+ \d{2},\d{4} at \d{2}:\d{2}:\d{2} UTC from IEEE Xplore\.\s*'
        r'Restrictions apply\.',  # Match the ending fixed phrase
        re.MULTILINE  # Enable multi-line matching
    )
    text = re.sub(footer_pattern, '', text)

    text = re.sub(r'\$\$\$\$', '\n\n\n', text)

    text = unicodedata.normalize('NFKD', text)

    text = text.encode('ascii', 'ignore').decode('ascii')

    text = ''.join(ch for ch in text if unicodedata.category(ch)
                   [0] not in ['C'])

    text = re.sub(r'/[a-zA-Z#]+\d+', '', text)

    text = re.sub(r'(?:/[a-zA-Z#]+\d+)+', '', text)

    text = text.replace('/', ' ').replace('#', ' ')

    text = re.sub(r'[^A-Za-z0-9\s.,;:!?\'\"()-]', ' ', text)

    text = re.sub(r'-\n\s*', '', text)

    text = re.sub(r' {2,}', ' ', text)

    text = text.strip()

    return text


def extract_text_from_pdf(pdf_path):
    """
    The main extraction function that uses pymupdf to extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
    """
    text = ""
    try:
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + '$$$$'
        clean = clean_text(text)
        return clean
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def save_texts_to_json(pdf_texts: dict, output_json: str):
    """
    Saves the extracted texts to a JSON file.

    Args:
        pdf_texts (dict): Dictionary containing PDF file names and their extracted texts.
        output_json (str): Path to the output JSON file.
    """
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(pdf_texts, file, indent=4, ensure_ascii=False)


def main():
    """
    The main function to scrapp the PDF files and save the extracted texts to JSON files.
    """
    current_file_path = os.path.abspath(__file__)
    ieee_pdf_folder = os.path.join(
        os.path.dirname(current_file_path), 'ieee_pdfs')
    elsevier_pdf_folder = os.path.join(
        os.path.dirname(current_file_path), 'elsevier_pdfs')
    ieee_output_json = os.path.join(os.path.dirname(
        current_file_path), 'ieee_pdf_texts.json')
    elsevier_output_json = os.path.join(os.path.dirname(
        current_file_path), 'elsevier_pdf_texts.json')

    ieee_pdf_texts = extract_texts_from_folder(ieee_pdf_folder)
    elsevier_pdf_texts = extract_texts_from_folder(elsevier_pdf_folder)

    save_texts_to_json(ieee_pdf_texts, ieee_output_json)
    save_texts_to_json(elsevier_pdf_texts, elsevier_output_json)


if __name__ == "__main__":
    main()
