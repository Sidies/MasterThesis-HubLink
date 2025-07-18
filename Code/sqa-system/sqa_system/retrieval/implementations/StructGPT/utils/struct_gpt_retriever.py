from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqa_system.core.data.models import Triple
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self, args, graph: KnowledgeGraph):
        self.args = args
        # self.initialize_PLM(args)
        # self.initialize_KG(args)
        self.cur_ents = set()
        self.graph = graph
        self.max_workers = args.max_workers

    def get_retrieval_information(self, iteration: int, gold_relations=None, all_rel_one_hop: List[Triple] = None, bidirectional=False):
        """
        NEW: We updated the function to work with the Knowledge Graph class of the
        SQA system.
        """
        triples_per_hop = {}
        new_entities = []
        if not all_rel_one_hop:
            all_rel_one_hop =  []
        logger.debug(f"Getting retrieval information for {len(self.cur_ents)} entities")
        for entity_id in self.cur_ents:
            entity = self.graph.get_entity_by_id(entity_id)
            if not entity:
                # possibly a Literal
                continue
            
            # Outgoing relations (entity as head)
            for relation in all_rel_one_hop:
                if gold_relations and relation.predicate not in gold_relations:
                    continue
                # Construct triple as [subject, predicate, object]
                triples_per_hop.setdefault(iteration, set()).add(relation)
                if relation.entity_subject.uid in self.cur_ents:
                    new_entities.append(relation.entity_object.uid)
                elif bidirectional:
                    new_entities.append(relation.entity_subject.uid)
                    
        # Convert set to list
        self.reset_cur_ents(new_entities)
        return {iteration: list(triples_per_hop[iteration])}, new_entities
        

    def get_retrieval_relations(self, first_flag=False, bidirectional=False):
        """
        NEW: Here we parallelize the retrieval of relations for each entity to make
        the retrieval faster
        """
        rels = []
        logger.debug(f"Getting retrieval relations for {len(self.cur_ents)} entities")
        
        # Use ThreadPoolExecutor to parallelize entity processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Map each entity to its relations using a regular loop
            future_to_entity = {}
            for entity_id in self.cur_ents:
                future = executor.submit(self._get_relations_for_entitiy_id, entity_id, bidirectional)
                future_to_entity[future] = entity_id
            
            # Collect all relations
            for future in as_completed(future_to_entity):
                entity_rels = future.result()
                rels.extend(entity_rels)
                
        return rels
    
    def _get_relations_for_entitiy_id(self, entity_id, bidirectional=False):
        rels = []
        entity = self.graph.get_entity_by_id(entity_id)
        if not entity:
            # possibly a Literal
            return []
        # Outgoing relations
        outgoing = self.graph.get_relations_of_head_entity(entity)
        rels.extend(outgoing)
        # NEW: When bidirectional, also include incoming relation predicates.
        if bidirectional:
            incoming = self.graph.get_relations_of_tail_entity(entity)
            rels.extend(incoming)
        return rels
        
    def reset_cur_ents(self, entity_list):
        # ----> NEW
        # Changed to set to avoid duplicates
        self.cur_ents = set(entity_list)
        # <----
        # logger.debug("Current entity num: ", len(self.cur_ents))

    def update_cur_ents(self, filtered_triples_per_hop):
        """
        New: Updated to work with the Knowledge Graph class of the SQA system.
        """
        new_entities = set()
        for hop, triples in filtered_triples_per_hop.items():
            for tri in triples:
                # tri is in the format: [subject, predicate, new_entity]
                new_entities.add(tri[2])
        self.reset_cur_ents(list(new_entities))

    def extract_facts(self, facts, response):
        response = response.lower().strip()
        filtered_facts = []
        for tri in facts:
            h, r, t = tri
            if self.filter_relation(r):
                continue
            nor_r = self.normalize_relation(r)
            if nor_r in response:
                filtered_facts.append(tri)
        return filtered_facts

    def filter_relation(self, rel):
        # same criteria as GraftNet
        relation = rel
        if relation == "common.topic.notable_types": return False
        if relation == "base.kwebbase.kwtopic.has_sentences": return False
        domain = relation.split(".")[0]
        if domain == "type" or domain == "common": return True
        return False

    def should_ignore(self, rel):
        if self.filter_relation(rel):
            return True
        return False

    def normalize_relation(self, rel):
        # e.g. <fb:film.film.other_crew>
        rel_surface = rel
        # replace '.' and '_' with ' '
        rel_surface = rel_surface.replace('.', ' ')
        # only keep the last two words
        rel_surface = ' '.join(rel_surface.split(' ')[-2:])
        rel_surface = rel_surface.replace('_', ' ')
        return rel_surface

    def get_one_hop_cand_rels(self, question):
        pass

    def get_tails_list(self, cur_ents):
        tails = [self.graph.id2ent[ent] for ent in cur_ents]
        return tails