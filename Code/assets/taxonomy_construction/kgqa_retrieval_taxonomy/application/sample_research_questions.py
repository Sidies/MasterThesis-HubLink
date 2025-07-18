import os
import json
from typing import Dict
import pandas as pd
from sqa_system.core.config.models import DatasetConfig, LLMConfig
from sqa_system.core.data.extraction.paper_content_extractor import PaperContentExtractor, PaperContent
from sqa_system.core.data.dataset_manager import DatasetManager
from sqa_system.core.data.models import Publication


# ---- HOW TO USE THIS SCRIPT ----
# First and foremost, this script is designed for our thesis data and will only work
# with the dataset we have. It is not a general-purpose script.
# This script can be rerun to reproduce the exact list of research questions that we
# used in our thesis for the example application of the taxonomy.
#
# REQUIREMENTS:
# - You need to have the SQA System installed
#
# INSTRUCTIONS:
# 1. Just run the script. It will load the dataset, extract the research questions,
#    categorize the papers, and sample the research questions.
# 2. The sampled research questions will be printed to the console and saved to a
#    JSON file named 'sampled_research_questions.json' in the same directory as this script.
# ---- END OF HOW TO USE THIS SCRIPT ----

# For reproducibility, we set a random seed to the
# publication year of the thesis
RANDOM_SEED = 2025
# This is the sample size of research questions that we want
# to sample from the dataset
SAMPLE_SIZE = 20


def main():
    """Main function to run the sampling of research questions."""
    # First we load the dataset and the research questions
    combined_df, publication_dataset = load_ecsa_icsa_dataset()
    research_questions_dict = extract_research_questions(
        publication_dataset.get_all_entries())
    # Then we parse the taxonomy fields and add them to the DataFrame
    taxonomy_data = combined_df.apply(
        parse_taxonomy_fields, axis=1, result_type='expand')
    extended_df = pd.concat([combined_df, taxonomy_data], axis=1)
    # Remove duplicate columns
    extended_df = extended_df.loc[:, ~
                                  extended_df.columns.duplicated(keep='first')]
    # Next we merge the DataFrame with the research questions
    merged_df = merge_papers_with_rqs(extended_df, research_questions_dict)
    # Only keep entries with a matching research question
    merged_df = merged_df[merged_df['RQ'].apply(lambda lst: len(lst) > 0)]
    # Explode the research questions into separate rows
    exploded = merged_df.explode('RQ').rename(
        columns={'RQ': 'ResearchQuestion'})

    # Finally we categorize the papers for sampling
    categorized_df = categorize_for_sampling(exploded)

    # And do the stratified sampling
    sampled_df = do_stratified_sampling(categorized_df, SAMPLE_SIZE)

    print_research_questions(sampled_df)
    write_research_questions_to_file(
        sampled_df, 'sampled_research_questions.json')


def do_stratified_sampling(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    This function does the main logic for the sampling of questions. 
    It tries to sample the same number of questions from each category.
    If there are too much questions in a category, it randomly samples
    from that category.
    """
    num_categories = len(df['SamplingCategory'].unique())
    # Get the amount of samples per category
    # Note we oversample here, and reduce afterwards
    samples_per_category = max(1, sample_size // num_categories) + 1

    sampled_rows = []
    for cat, subset in df.groupby('SamplingCategory'):
        if len(subset) <= samples_per_category:
            sampled_rows.append(subset)
        else:
            sampled_rows.append(subset.sample(
                samples_per_category, random_state=RANDOM_SEED))

    sampled_df = pd.concat(sampled_rows)
    # Now we reduce the sample size to the desired size
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(sample_size, random_state=RANDOM_SEED)
    return sampled_df.reset_index(drop=True)


def load_ecsa_icsa_dataset() -> pd.DataFrame:
    """
    Loads the dataset with the publications from the file and further parses
    the annotations fields.
    """
    publications_config = DatasetConfig.from_dict({
        "additional_params": {},
        "file_name": "merged_ecsa_icsa.json",
        "loader": "JsonPublicationLoader",
        "loader_limit": -1
    })
    publication_dataset = DatasetManager().get_dataset(publications_config)

    print(
        f"Loaded {len(publication_dataset.get_all_entries())} publications from the JSON file.")

    publication_df = publication_dataset.to_dataframe()

    additional_fields_dict = publication_df['additional_fields'].to_dict()
    additional_fields_df = pd.DataFrame.from_dict(
        additional_fields_dict, orient='index')

    annotations_dict = additional_fields_df['annotations'].to_dict()
    annotations_df = pd.DataFrame.from_dict(annotations_dict, orient='index')

    metadata_dict = annotations_df['Meta Data'].to_dict()
    metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')

    # Concatenate everything into one DataFrame
    combined_df = pd.concat(
        [publication_df, additional_fields_df, annotations_df, metadata_df], axis=1)

    print("Columns in combined DataFrame:")
    print(combined_df.columns.tolist())

    return combined_df, publication_dataset


def extract_research_questions(publications: list[Publication]) -> dict[str, list[str]]:
    """
    Extracts the research questions from the Publications.

    Args:
        publications (list[Publication]): List of publications from which to extract
            the research questions.

    Returns:
        dict[str, list[str]]: A dictionary where the keys are DOIs and the values
            are lists of research questions.    
    """
    llm_config_extraction = LLMConfig.from_dict({
        "additional_params": {},
        "endpoint": "OpenAI",
        "name_model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_tokens": -1
    })

    # Run the extraction:
    summaries: Dict[str, PaperContent] = {}
    extractor = PaperContentExtractor(llm_config=llm_config_extraction)
    for publication in publications:
        summaries[publication.doi] = extractor.extract_paper_content(
            publication=publication,
            update_cache=False
        )

    # Prepare the research questions
    has_no_fulltext = 0
    has_no_research_questions = 0
    research_questions = {}
    for doi, summary in summaries.items():
        if not summary:
            print(f"Paper with DOI {doi} has no fulltext.")
            has_no_fulltext += 1
            research_questions[doi] = ""
            continue
        if summary.research_questions:
            research_questions[doi] = summary.research_questions
        else:
            print(f"Paper with DOI {doi} verbalizes no research questions.")
            has_no_research_questions += 1
            research_questions[doi] = ""

    print("-----------------------------")
    print(f"Number of papers without fulltext: {has_no_fulltext}")
    print(
        f"Number of papers without research questions: {has_no_research_questions}")

    research_question_texts = {}
    number_of_research_questions = 0
    for doi, questions in research_questions.items():
        research_question_texts[doi] = [q.text for q in questions]
        number_of_research_questions += len(questions)

    print(
        f"Total number of research questions extracted: {number_of_research_questions}")
    return research_question_texts


def parse_taxonomy_fields(row: pd.Series) -> dict:
    """
    For each row in the combined DataFrame, parse out the relevant
    taxonomy fields.
    """
    parsed = {}

    parsed['PaperClass'] = row.get('Paper class', None)
    parsed['ResearchLevel'] = row.get('Research Level', None)
    parsed['Kind'] = row.get('Kind', None)

    validity_data = row.get('Validity', {})
    if isinstance(validity_data, dict):
        parsed['InputData'] = validity_data.get('Input Data')
        parsed['ToolSupport'] = validity_data.get('Tool Support')
        parsed['ReplicationPackage'] = validity_data.get('Replication Package')
        parsed['ThreatsToValidity'] = validity_data.get('Threats To Validity')

    first_research_object = row.get('First Research Object', {})
    if isinstance(first_research_object, dict):
        parsed['ResearchObject'] = first_research_object.get('Research Object')
        first_eval = first_research_object.get('First Evaluation', {})
        parsed['EvaluationMethod'] = first_eval.get('Evaluation Method')
        parsed['Properties'] = first_eval.get('Properties')

    return parsed


def merge_papers_with_rqs(combined_df, research_questions_dict):
    """
    Merges the combined DataFrame with the research questions dictionary.
    """
    df = combined_df.copy()
    df['RQ'] = (
        df['doi']
        .map(lambda doi: research_questions_dict.get(doi, []))
        .apply(lambda x: x if isinstance(x, list) else [x])
    )
    return df


def categorize_for_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes papers for sampling based on their class and research level.
    """
    def make_category(row):
        return f"{row['PaperClass'] or ''} | {row['ResearchLevel'] or ''}"

    df['SamplingCategory'] = df.apply(make_category, axis=1)
    return df


def print_research_questions(df: pd.DataFrame):
    """Pretty prints the research questions for each paper."""
    grouped = (
        df
        .groupby(['title', 'doi', 'SamplingCategory'])['ResearchQuestion']
        .apply(list)
        .reset_index()
    )
    for paper_idx, row in enumerate(grouped.itertuples(index=False), start=1):
        title = row.title
        doi = row.doi
        category = row.SamplingCategory
        rqs = row.ResearchQuestion

        print(f"--- Paper {paper_idx} ---")
        print(f"Title: {title}")
        print(f"DOI: {doi}")
        print(f"Sampling Category: {category}")
        print("RQs:")
        for i, rq in enumerate(rqs, start=1):
            print(f"  {i}. {rq}")
        print()


def write_research_questions_to_file(df: pd.DataFrame, output_path: str):
    """Writes all research questions into a .json file."""
    json_data = []

    for _, row in df.iterrows():
        json_data.append({
            "research_question": row['ResearchQuestion'],
            "title": row['title'],
            "doi": row['doi'],
            "annotations": []
        })

    current_directory = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(current_directory, output_path)

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Output saved to '{full_path}'")


if __name__ == "__main__":
    main()
