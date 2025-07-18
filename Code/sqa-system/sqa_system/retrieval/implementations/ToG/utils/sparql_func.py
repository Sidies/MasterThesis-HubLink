"""
The following file is a implementation of the Think-on-Graph retriever.
The repository can be found here: https://github.com/IDEA-FinAI/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67

Their code has been adapted to work with this project.
"""
import json
import time
from typing import Any, Dict, List, Tuple
import openai
import re
from pprint import pformat
from SPARQLWrapper import SPARQLWrapper, JSON
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.core.data.models.triple import Triple
from .prompt_list import *
from .utils import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


    
def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id, graph:KnowledgeGraph):
    """
    New: We adapted the original code as the ID to entity name conversion
    is now done by the KnowledgeGraph class.
    """
    if not graph.validate_graph_connection():
        raise ValueError("No connection")
    name = graph.get_entity_by_id(entity_id)
    
    if name is None or name == "":
        return "UnName_Entity"
    else:
        return name.text

def clean_relations(
    string: str,
    entity_id: str,
    tail_relations: List[str],
    head_relations: List[str]
) -> Tuple[bool, List[Dict[str, Any]] | str]:
    """
    New: We decided to rework the original parser here as we found
    it to not be working with many of the LLM outputs we received.
    This is because the LLM outputs are not always in the same format.
    Our adapted implementation is more robust and works with
    many different formats.
    """
    
    if not isinstance(string, str):
        return False, "No relations found"

    tail_relations_lower = set(rel.lower() for rel in tail_relations)
    head_relations_lower = set(rel.lower() for rel in head_relations)
    
    patterns = [
        # Original pattern
        r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}",  
        # Pattern without curly braces
        r"(?P<relation>[^():]+)\s*\(Score:\s*(?P<score>[0-9.]+)\)",  
        # Simple "relation: score" pattern    
        r"(?P<relation>[^():]+):\s*(?P<score>[0-9.]+)",     
         # Reversed "score: relation" pattern           
        r"(?P<score>[0-9.]+)[:]\s*(?P<relation>[^():]+)"              
    ]
    
    relations = []
    found_relations = set()

    for pattern in patterns:
        for match in re.finditer(pattern, string, re.IGNORECASE | re.MULTILINE):
            relation = match.group("relation").strip()
            score = match.group("score").strip()
            
            if ';' in relation:
                continue
            
            if not relation or not score:
                continue
            
            try:
                score = float(score)
            except ValueError:
                continue
            
            original_relation = relation
            # Remove "wiki.relation." prefix if present
            relation = relation.replace("wiki.relation.", "")
            relation_lower = relation.lower()
            
            if relation_lower in found_relations:
                continue
            
            if relation_lower in tail_relations_lower:
                relations.append({
                    "entity": entity_id,
                    "relation": original_relation,
                    "score": score,
                    "head": False
                })
                found_relations.add(relation_lower)
            elif relation_lower in head_relations_lower:
                relations.append({
                    "entity": entity_id,
                    "relation": original_relation,
                    "score": score,
                    "head": True
                })
                found_relations.add(relation_lower)
            else:
                continue
    
    # If no relations found, try a last-resort method
    if not relations:
        lines = string.split('\n')
        for line in lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                relation = parts[0].strip()
                score_match = re.search(r'[0-9.]+', parts[1])
                if score_match:
                    score = float(score_match.group())
                    original_relation = relation
                    relation = relation.replace("wiki.relation.", "")
                    relation_lower = relation.lower()
                    
                    if relation_lower in found_relations:
                        continue
                    
                    if relation_lower in tail_relations_lower:
                        relations.append({
                            "entity": entity_id,
                            "relation": original_relation,
                            "score": score,
                            "head": False
                        })
                        found_relations.add(relation_lower)
                    elif relation_lower in head_relations_lower:
                        relations.append({
                            "entity": entity_id,
                            "relation": original_relation,
                            "score": score,
                            "head": True
                        })
                        found_relations.add(relation_lower)
                    else:
                        continue
    
    if not relations:
        return False, "No relations found"
    
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def construct_relation_prune_prompt(question, entity_name, total_relations, width):
    return extract_relation_prompt % (width, width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id:str, 
                          entity_name:str, 
                          pre_relations:List[str], 
                          pre_head, 
                          question:str, 
                          technique:str,
                          width:int,
                          temperature_reasoning:int,
                          model_name:str,
                          graph:KnowledgeGraph,
                          llm_config:LLMConfig,
                          bidirectional:bool,
                          replace_contribution:bool):
    """
    NEW: The original code was adapted to work with the new KnowledgeGraph class.
    We kept the logic of the original code the same.
    """
    
    # get head and tail relations
    entity_knowledge = graph.get_entity_by_id(entity_id)
    if entity_knowledge is None:
        return []
    head_neighbors = graph.get_relations_of_head_entity(entity_knowledge)
    if bidirectional:
        tail_neighbors = graph.get_relations_of_tail_entity(entity_knowledge)
    else:
        tail_neighbors = []
    
    # transform to only the descriptions of the relation
    head_relations = {}
    for relation in head_neighbors:
        if relation.predicate not in head_relations:
            head_relations[relation.predicate] = []
        head_relations[relation.predicate].append(relation)
    tail_relations = {}
    for relation in tail_neighbors:
        if relation.predicate not in tail_relations:
            tail_relations[relation.predicate] = []
        tail_relations[relation.predicate].append(relation)
    
    total_relations = list(head_relations.keys()) + list(tail_relations.keys())
    total_relations.sort(key=str)  # make sure the order in prompt is always equal
    
    if replace_contribution:
        total_relations = replace_string(total_relations, "contribution", "paper content")

    if len(total_relations) <= 0:
        logger.debug("No further relations found for entity %s", entity_id)
        return []
    
    logger.debug("The extracted relations for entity %s are %s", entity_id, total_relations)
    
    if technique == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, width)
        # logger.debug("Asking LLM which relations are relevant: %s", prompt)

        result = run_llm(prompt, temperature_reasoning, llm_config)
        if "human_language.main_country" in str(result):
            logger.warning("The ToG retriever hallucinated on evaluating the relations.")
        logger.debug("The LLM returned: %s", result)
        logger.debug("Preparing relations...")
        if replace_contribution:
            result = replace_string(result, "paper content", "contribution")
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, tail_relations, head_relations) 
        formatted_output = pformat(retrieve_relations_with_scores, indent=2, width=100)
        logger.debug("The relations with scores are: %s", formatted_output)        

    elif technique == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, width)
        if len(topn_relations) <= 0:
            return []
        if replace_contribution:
            topn_relations = replace_string(topn_relations, "paper content", "contribution")
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
        formatted_output = pformat(retrieve_relations_with_scores, indent=2, width=100)
        logger.debug("The relations with scores are: %s", formatted_output)        
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        # add the relations
        for entry in retrieve_relations_with_scores:
            if entry["head"] and entry["relation"] in head_relations:
                entry["triples"] = head_relations[entry["relation"]]
            elif not entry["head"] and entry["relation"] in tail_relations:
                entry["triples"] = tail_relations[entry["relation"]]
            else:
                logger.warning(f"Could not add triples for relation {entry['relation']}")
            
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
    
def entity_search(entity, relation, graph:KnowledgeGraph, head=True):
    """
    NEW: The function has been adapted to work with the new KnowledgeGraph class,
    while keeping the logic of the original code.
    """
    if not graph.validate_graph_connection():
        raise ConnectionError("Failed to connect to ORKG graph endpoint.")
    
    entities = []
    if head:
        tail_entities_extract = graph.get_relations_of_head_entity(graph.get_entity_by_id(entity))
        for relation_extract in tail_entities_extract:
            if relation_extract.predicate == relation:
                entities.append(relation_extract.entity_object.uid)
        
    else:
        head_entities_extract = graph.get_relations_of_tail_entity(graph.get_entity_by_id(entity))
        for relation_extract in head_entities_extract:
            if relation_extract.predicate == relation:
                entities.append(relation_extract.entity_subject.uid)
    
    return entities

def replace_string(string: List[str] | str, replace:str, replace_with: str) -> List[str] | str:
        """
        Replaces the string in str_list with replace_with.
        
        Args:
            string: The string or strings to be replaced
            replace: String to be replaced
            replace_with: String to replace with
        """
        if isinstance(string, str):
            return string.replace(replace, replace_with)
        str_list = string
        for i, s in enumerate(str_list):
            if replace in s:
                str_list[i] = s.replace(replace, replace_with)
        return str_list

def entity_score(question, entity_candidates_id, score, relation, width, temperature_exploration, llm_config, technique, graph):
    
    entity_candidates = [id2entity_name_or_type(entity_id, graph) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if technique == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)
        # ---> NEW
        # Added log
        logger.debug("Asking LLM to score entities")

        result = run_llm(prompt, temperature_exploration, llm_config)
        logger.debug("The LLM returned: %s", result)
        # <---
        # ----> NEW
        # Added a check. Without this check the execution can crash.
        if result is None:
            return [], entity_candidates, entity_candidates_id
        else:
        # <----
            return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    elif technique == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, total_triples):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    # ---> NEW
    # Added triples to the list for later retrieved context output
    candidate_triples = [entity['triples']] * len(entity_candidates)
    # <---
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    # ---> NEW
    # Added triples to the list for later retrieved context output
    total_triples.extend(candidate_triples)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, total_triples
    # <---



def half_stop(question, cluster_chain_of_entities, depth, temperature_reasoning, llm_config) -> str:
    """
    NEW: Updated to work with the new parameters and changed to logging.
    """
    logger.debug("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, temperature_reasoning, cluster_chain_of_entities, llm_config)
    # print(question, answer, cluster_chain_of_entities)
    return str(answer)


def generate_answer(question, temperature_reasoning, cluster_chain_of_entities, llm_config): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    # ---> NEW
    # Changed parameters
    result = run_llm(prompt, temperature_reasoning, llm_config)
    # <---
    return result


def entity_prune(total_triples, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, graph, width=3):
    """
    NEW: Only a minor change to the original code. We added the triples to the
    lists for later retrieved context output.
    """
    zipped = list(zip(total_triples, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[6], reverse=True)
    sorted_triples, sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped], [x[6] for x in sorted_zipped]

    triples, entities_id, relations, candidates, topics, heads, scores = sorted_triples[:width], sorted_entities_id[:width], sorted_relations[:width], sorted_candidates[:width], sorted_topic_entities[:width], sorted_head[:width], sorted_scores[:width]
    merged_list = list(zip(triples, entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(trip, id, rel, ent, top, hea, score) for trip, id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], [], []
    triples, entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id, graph) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i], triples[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads, triples


def reasoning(question, temperature_reasoning, llm_config, cluster_chain_of_entities):
    formatted_output = pformat(cluster_chain_of_entities, indent=2, width=100)
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    # ---> NEW
    # Added log
    logger.debug("Asking LLM")

    response = run_llm(prompt, temperature_reasoning, llm_config)
    
    logger.debug("The LLM returned: %s", response)
    
    result = extract_answer(response)
    
    logger.debug("The extracted answer is %s", result)
    # <----
    if if_true(result):
        return True, response
    else:
        return False, response
    


