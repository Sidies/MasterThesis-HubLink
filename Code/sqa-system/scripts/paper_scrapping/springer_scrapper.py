import requests
import os
import time
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def get_fulltext(doi):
    """
    Retrieves the fulltext from the given URL using the doi
    """
    url = f"https://link.springer.com/content/pdf/{doi}.pdf"
    headers = {
        'Accept': 'application/pdf'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:  # Rate limit
                wait_time = random.uniform(1, 3)
                time.sleep(wait_time)
                continue
            else:
                logging.error("Failed to download %s. Status code: %s", doi, response.status_code)
                return None
        except Exception as e:
            logging.error("Error downloading %s: %s", doi, str(e))
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
                continue
            return None
    return None

def sanitize_filename(doi):
    """Replace invalid filename characters with underscores"""
    return doi.replace('/', '_').replace('\\', '_')

def main(dois, output_dir):
    for doi in tqdm(dois):
        doi = doi.replace("\\_", "_")
        file_name = f"{sanitize_filename(doi)}.pdf"
        save_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(save_path):
            logging.info(f"PDF already exists for {doi}")
            continue
            
        pdf_content = get_fulltext(doi)
        if pdf_content:
            try:
                with open(save_path, 'wb') as f:
                    f.write(pdf_content)
                logging.info(f"Successfully downloaded {doi}")
                time.sleep(random.uniform(1, 2))  # Polite delay between requests
            except Exception as e:
                logging.error(f"Error saving PDF for {doi}: {str(e)}")
        else:
            logging.error(f"Failed to retrieve PDF for {doi}")

if __name__ == '__main__':
    # We load the dois from the missing_fulltext_dois.txt file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    missing_dois_path = os.path.join(current_file_dir, 'missing_fulltext_dois.txt')
    with open(missing_dois_path, 'r') as f:
        dois = f.read().splitlines()
    
    output_dir = os.path.join(current_file_dir, 'elsevier_pdfs')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    main(dois, output_dir)
