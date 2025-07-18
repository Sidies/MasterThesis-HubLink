"""
We used this script to combine the different exports into one file.
"""
import json
import os
from typing import List
from pybtex.database import parse_file
import pandas as pd

current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(root_path, "data", "external")
output_path = os.path.join(data_path, "merged_ecsa_icsa.json")


def load_bib():
    """
    Load and parse the .bib file using pybtex.  
    """
    bib_path = os.path.join(data_path, "ECSA-ICSA-Proceedings.bib")
    print(f"Loading .bib file {bib_path}...")
    bib_data = parse_file(bib_path)
    bib_entries = {}

    for _, entry in bib_data.entries.items():
        fields = entry.fields
        doi = fields.get('doi', '').strip().lower().replace("\\_", "_")

        if doi:
            key = doi
        else:
            # Skip entries without DOI
            continue

        # Convert authors from list of Person objects to a single string
        authors = ' and '.join([' '.join(person.get_part('first') + person.get_part('last'))
                                for person in entry.persons.get('author', [])])
        fields['author'] = authors
        # check if the key already exists
        if key in bib_entries:
            print(f"Duplicate key found: {key}")
            continue
        bib_entries[key] = fields

    return bib_entries

def load_evidence_values():
    """
    Load and parse the evidence values xlsx file.
    
    Because here only the Replication Package Link is needed, we only
    parse those values.
    """
    evidence_values_path = os.path.join(
        data_path, "evidence-values.xlsx")
    print(f"Loading Evidence Values file {evidence_values_path}...")
    evidence_values = pd.read_excel(evidence_values_path, sheet_name=None)
    table = evidence_values['Tabelle1']
    
    table_dict = table.to_dict(orient='list')
    
    links = table_dict['Replication Package Link']
    dois = table_dict['DOI']
    
    evidence_data = {}
    for link, doi in zip(links, dois):
        # if link is not NaN
        if pd.notna(link):
            evidence_data[doi.strip().lower()] = {
                "replication_package_link": link
            }
    return evidence_data


def load_json_ecsa_extract():
    """
    Load and parse the ECSA JSON file.
    """
    springerlink_json_path = os.path.join(
        data_path, "export_SpringerLink_ecsa_export.json")
    print(f"Loading JSON file {springerlink_json_path}...")
    with open(springerlink_json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    json_entries = {}
    for paper in data.get('papers', []):
        doi = paper.get('doi', '').strip().lower()

        if doi:
            key = doi
        else:
            # Skip entries without DOI
            continue
        # check if the key already exists
        if key in json_entries:
            print(f"Duplicate key found: {key}")
            continue
        # we add missing information
        paper['publisher'] = 'SpringerLink'
        paper['venue_name'] = 'European Conference on Software Architecture (ECSA)'
        paper['venue_type'] = 'Conference'
        json_entries[key] = paper

    return json_entries


def load_json_ieee_extract():
    """
    Load and parse the IEEE JSON file.
    """
    json_path_ieee = os.path.join(os.path.dirname(
        current_file_path), "ieee_pdf_texts.json")
    article_mapping = os.path.join(os.path.dirname(
        current_file_path), "ieee_pdfs", "article_to_doi_mapping.json")
    print(f"Loading JSON file {json_path_ieee}...")
    with open(json_path_ieee, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(article_mapping, 'r', encoding='utf-8') as json_file:
        article_to_doi_mapping = json.load(json_file)

    json_entries = {}
    for key, value in data.items():
        article_number = key.replace('.pdf', '')
        doi = article_to_doi_mapping.get(article_number)
        if not doi or doi == "":
            continue
        json_entries[doi.lower()] = {
            "fulltext": value,
            "publisher": "IEEE",
            "venue_name": "International Conference on Software Architecture (ICSA)",
            "venue_type": "Conference"
        }

    return json_entries


def load_json_elsevier_extract():
    """
    Load and parse the JSON Elsevier file.

    Args:
        json_path (str): Path to the JSON file.
    """
    json_path_elsevier = os.path.join(os.path.dirname(
        current_file_path), "elsevier_pdf_texts.json")
    print(f"Loading JSON file {json_path_elsevier}...")
    with open(json_path_elsevier, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    json_entries = {}
    for key, value in data.items():
        json_entries[key.lower()] = {
            "fulltext": value,
            "publisher": "Springer",
            "venue_name": "European Converence Software Architecture (ECSA)",
            "venue_type": "Journal"
        }

    return json_entries


def merge_entries(bib_entries, json_entries: List[dict]):
    """
    Merge bib and JSON entries based on DOI or title.
    The bib entries are used as the base, and JSON entries are merged into them.

    Args:
        bib_entries (dict): Dictionary of bib entries keyed by DOI or title.
        json_entries (List[dict]): List of Dictionary of JSON entries keyed by DOI or title.
    """
    merged = {}
    all_keys = set(bib_entries.keys())

    for key in all_keys:
        merged_entry = {}
        key = key.lower().replace("\\_", "_")

        bib_entry = bib_entries.get(key, {})
        for field, value in bib_entry.items():
            merged_entry[field] = value
        for entry in json_entries:
            if key in entry:
                json_entry = entry[key]
                for field, value in json_entry.items():
                    merged_entry[field] = value

        merged[key] = merged_entry

    # if we have two types of author keys, we only keep the one from the bib file
    for entry in merged.values():
        if 'authors' in entry and 'author' in entry:
            del entry['authors']

    return list(merged.values())


def check_missing_keys(merged_entries, bib_entries):
    """
    Check if all keys from the bib file are present in the merged file.
    """
    merged_keys = {entry['doi'].lower().replace("\\_", "_")
                   for entry in merged_entries}
    bib_keys = set(key.lower().replace("\\_", "_")
                   for key in bib_entries)
    missing_keys = []
    for key in bib_keys:
        if key not in merged_keys:
            missing_keys.append(key)
    if missing_keys:
        print(
            f"Warning: {len(missing_keys)} keys are missing from the merged file.")
        print(f"Missing keys: {missing_keys}")
    else:
        print("All keys from the bib file are present in the merged file.")


def main():
    """
    The main merging script
    """

    bib_entries = load_bib()
    print(f"Loaded {len(bib_entries)} entries from .bib file.")
    
    evidence_values = load_evidence_values()
    print(f"Loaded {len(evidence_values)} entries from Evidence Values file.")

    json_entries = load_json_ecsa_extract()
    print(f"Loaded {len(json_entries)} entries from JSON file.")

    json_entries_2 = load_json_ieee_extract()
    print(f"Loaded {len(json_entries_2)} entries from JSON file.")
    json_entries_3 = load_json_elsevier_extract()

    print("Merging entries...")
    merged_entries = merge_entries(
        bib_entries, [json_entries, json_entries_2, json_entries_3, evidence_values])
    print(f"Merged into {len(merged_entries)} total entries.")

    print(f"Writing merged data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump({'papers': merged_entries}, out_file, indent=4)

    check_missing_keys(merged_entries, bib_entries)

    # now we check which bib entries have no fulltext
    missing_fulltext = []
    for entry in merged_entries:
        if not entry.get('fulltext'):
            missing_fulltext.append(entry['doi'])
    if missing_fulltext:
        print(f"Warning: {len(missing_fulltext)} entries are missing fulltext.")
        missing_dois_path = os.path.join(os.path.dirname(
            current_file_path), 'missing_fulltext_dois.txt')
        with open(missing_dois_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(missing_fulltext))
        print(f"List of missing DOIs written to: {missing_dois_path}")

    print("Merging completed successfully.")


if __name__ == '__main__':
    main()
