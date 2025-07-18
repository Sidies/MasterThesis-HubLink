"""
The following file is a implementation of the Think-on-Graph retriever.
The repository can be found here: https://github.com/IDEA-FinAI/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67

Their code has been adapted to work with this project.
"""

from typing import List
from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.data.models.dataset.implementations.qa_dataset import QADataset
from .prompt_list import *
import json
import time
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.language_model.enums.llm_enums import EndpointType
from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.data.dataset_manager import DatasetManager
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """
    # ----> NEW
    if not corpus:
        return [], []
    # <----

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    # ----> NEW
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:width]
    top_relations = [corpus[i] for i in top_n_indices]
    top_scores = [doc_scores[i] for i in top_n_indices]

    return top_relations, top_scores
    # <----



def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

def run_llm(prompt, temperature, llm_config:LLMConfig):
    """
    New: We adapted the whole function to work with our
    SQA system implementation.
    """
    llm_provider = LLMProvider()
    
    llm_config.temperature = temperature
    
    # Get the LLM adapter
    try:
        llm_adapter = llm_provider.get_llm_adapter(llm_config)
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
        return None
    except APIKeyMissingError:
        logger.error("Error: OpenAI API key is missing")
        return None
    
    # NEW: We encountered an issue where the context window was too large
    # and the LLM would not respond. To avoid this, we reduce the prompt size
    max_length = 100000
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    
    # Prepare the messages
    messages = [
        {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user", "content": prompt}
    ]
    
    # Run the LLM
    max_retries = 3
    for _ in range(max_retries):
        try:
            response = llm_adapter.llm.invoke(messages)
            if isinstance(response, str):
                return response
            else:
                return response.content
        except Exception as e:
            logger.warning(f"LLM error: {str(e)}, retrying...")
            time.sleep(2)
    
    logger.debug("Max retries reached. Failed to get a response from the LLM.")
    return None

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates
    
    

def clean_scores(string: str, entity_candidates: List[str]) -> List[float]:
    """
    NEW: The whole function was overhauled by us as we encountered 
    issues where the LLM would not provide the desired output which
    makes the original parser fail. Because this happens extraordinarly
    often and renders open source LLMs unuseable, we decided to completly
    write a new parser that is more forgiving.
    """
    # Create a regex pattern for the score format
    score_pattern = r'(1\.0|0\.[0-9])'
    
    # Create a dictionary to store the highest score for each entity
    entity_scores = {entity: 0.0 for entity in entity_candidates}
    
    # Find all matches of entities and their scores
    for entity in entity_candidates:
        escaped_entity = re.escape(entity)
        pattern = rf'(?:\*\*|\()?{escaped_entity}(?:\*\*|\))?\s*:(?:(?!(?:\*\*|\()?(?:{"|".join(map(re.escape, entity_candidates))}))[\s\S])*?{score_pattern}'
        matches = re.findall(pattern, string, re.IGNORECASE)
        
        # If matches are found, update the score if it's higher than the current one
        if matches:
            highest_score = max(float(score) for score in matches)
            entity_scores[entity] = max(entity_scores[entity], highest_score)

    # If no matches are found, try an alternative
    # This is the answer string the authors of ToG expects the LLM to give
    if all(score == 0.0 for score in entity_scores.values()):
        pattern = r'(\d+\.\d+(?:,\s*\d+\.\d+)*)'
        matches = re.findall(pattern, string)
        
        if matches:
            scores = [float(score.strip()) for score in matches[0].split(',')]
            
            if len(scores) == len(entity_candidates):
                for entity, score in zip(entity_candidates, scores):
                    entity_scores[entity] = score

    # check if the sum of the scores is 1
    if sum(entity_scores.values()) == 1:
        return [entity_scores[entity] for entity in entity_candidates]
    else:
        return [1/len(entity_candidates)] * len(entity_candidates)


def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
def extract_answer(text):
    # ----> NEW
    # In some cases the text is None which crashes
    # the retrieval process. We added a check for this.
    if not text:
        return ""
    # <----
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

# ----> NEW
# Small change of the parameters
def generate_without_explored_paths(question, temperature_reasoning, llm_config:LLMConfig):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, temperature_reasoning, llm_config)
    return response
# <----


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(config:DatasetConfig):
    """
    New: We changed the dataset loading to the SQA system
    implementation. 
    """
    dataset = DatasetManager().get_dataset(config)
    if not isinstance(dataset, QADataset):
        raise ValueError("Dataset is not a QADataset.")
    
    qa_pairs = dataset.get_all_entries()
    
    return qa_pairs

def print_results(question, answer, cluster_chain_of_entities):
    """
    New: Prints the results to the log file.
    """
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    logger.debug("The ToG Retriever has returned the following results:")
    logger.debug(json.dumps(dict))