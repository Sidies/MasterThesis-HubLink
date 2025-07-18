import json
import requests
import os
import time
import random
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_article_number(doi, api_key):
    """
    Retrieves the article number for a given DOI using the IEEE API.

    Args:
        doi (str): The DOI of the paper.
        api_key (str): Your IEEE API key.

    Returns:
        str or None: The article number if found, else None.
    """
    base_url = 'https://ieeexploreapi.ieee.org/api/v1/search/articles'
    params = {
        'doi': doi,
        'apikey': api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        if not articles:
            logging.warning(f"No articles found for DOI: {doi}")
            return None
        article_number = articles[0].get('article_number')
        if not article_number:
            logging.warning(f"No article number found for DOI: {doi}")
        return article_number
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching article number for DOI {doi}: {e}")
        return None

def download_pdf(article_number, download_dir):
    """
    Downloads the PDF for a given article number using direct HTTP request.

    Args:
        article_number (str): The article number.
        download_dir (str): Directory to save downloaded PDFs.

    Returns:
        bool: True if download successful, False otherwise.
    """
    url = f'https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={article_number}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200 and response.headers['Content-Type'] == 'application/pdf':
            save_path = os.path.join(download_dir, f"{article_number}.pdf")
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Successfully downloaded PDF for article {article_number}")
            return True
        else:
            logging.error(f"Failed to download PDF for article {article_number}. Status code: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"Error downloading PDF for article {article_number}: {str(e)}")
        return False

def main(dois, api_key, output_dir):
    """
    Main function to process DOIs and download corresponding PDFs.

    Args:
        dois (list): List of DOIs.
        api_key (str): Your IEEE API key.
        output_dir (str): Directory to save PDFs.
    """
    article_to_doi_mapping = {}
    
    for doi in tqdm(dois):
        logging.info(f"Processing DOI: {doi}")
        article_number = get_article_number(doi, api_key)
        if not article_number:
            logging.warning(f"Skipping DOI {doi} due to missing article number.")
            continue
        
        filename = f"{article_number}.pdf"
        save_path = os.path.join(output_dir, filename)
        if os.path.exists(save_path):
            logging.info(f"PDF already exists: {save_path}")
            continue
            
        if download_pdf(article_number, output_dir):
            article_to_doi_mapping[article_number] = doi
            # Save the mappings after each successful download
            with open(os.path.join(output_dir, 'article_to_doi_mapping.json'), 'w') as f:
                json.dump(article_to_doi_mapping, f, indent=4)
            
            time.sleep(random.uniform(1, 2))

if __name__ == '__main__':
    # We load the dois from the missing_fulltext_dois.txt file
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    missing_dois_path = os.path.join(current_file_dir, 'missing_fulltext_dois.txt')
    with open(missing_dois_path, 'r') as f:
        dois = f.read().splitlines()
    
    # Configuration
    api_key = input("Enter your IEEE Xplore API Key: ")
    
    output_dir = os.path.join(current_file_dir, 'ieee_pdfs')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    main(dois, api_key, output_dir)
