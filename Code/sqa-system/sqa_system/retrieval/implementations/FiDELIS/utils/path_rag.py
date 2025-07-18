import numpy as np
from typing import List 

from concurrent.futures import ThreadPoolExecutor, as_completed

from sqa_system.core.data.models import Knowledge
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.logging.logging import get_logger

from .llm_backbone import LLM_Backbone

logger = get_logger(__name__)


class Path_RAG():

    def __init__(self, graph: KnowledgeGraph, args):
        self.llm_backbone = LLM_Backbone(llm_config=args["llm_config"],
                                         embedding_config=args["embedding_config"])
        self.args = args
        self.graph = graph
        self.cached_entity_relations_head = {}
        self.cached_entity_relations_tail = {}
        self.cached_embeddings = {}

    def cos_simiarlity(self, a: np.array, b: np.array):
        """
        calculate cosine similarity between two vectors
        Parameters:
            a: np.array, representing a single vector
            b: np.array, shape (n_vectors, vector_length), representing multiple vectors
        """
        a = a.reshape(1, -1)
        dot_product = np.dot(a, b.T).flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b, axis=1)

        epsilon = 1e-9
        cos_similarities = dot_product / (norm_a * norm_b + epsilon)
        return cos_similarities

    def get_entity_edges(self, entity: Knowledge) -> tuple[List[dict], List[dict]]:
        """
        NEW: We adapted the code to work with the Knowledge Graph Interface of the
        SQA system.
        """
        edges = []
        neighbors = []

        if entity is None:
            return edges, neighbors

        relations = (
            self._get_relations_of_tail_with_cache(entity)
            + self._get_relations_of_head_with_cache(entity)
        )

        unique_edge_attrs = {}
        neighbor_entities: List[Knowledge] = []
        entity_to_relation_mapping = {}

        for relation in relations:
            predicate = relation.predicate
            if predicate not in unique_edge_attrs:
                unique_edge_attrs[predicate] = predicate

            # Get the entity whether its left or right
            neighbor_entity = (
                relation.entity_subject if relation.entity_object == entity
                else relation.entity_object
            )
            neighbor_entities.append(neighbor_entity)
            entity_to_relation_mapping[neighbor_entity.uid] = relation

        edge_attributes = list(unique_edge_attrs.keys())
        neighbor_texts = [n.text for n in neighbor_entities]
        
        edge_embeddings = self._embed_batch_with_cache(edge_attributes)
        neighbor_embeddings = self._embed_batch_with_cache(neighbor_texts)

        # Reconstruct the edges and neighbors lists with embeddings.
        for attr, emb in zip(edge_attributes, edge_embeddings):
            edges.append({
                "attribute": attr,
                "embedding": emb
            })

        for e, emb in zip(neighbor_entities, neighbor_embeddings):
            neighbors.append({
                "attribute": e,
                "embedding": emb,
                "triple": entity_to_relation_mapping[e.uid]
            })

        return edges, neighbors
    
    def _embed_batch_with_cache(self, texts: List[str]) -> List[np.array]:
        """
        NEW:
        Embed a batch of texts with caching while making sure that the order of the
        embeddings is the same as the order of the texts.
        """
        # First identify which texts need to be embedded
        need_to_embed = []
        
        for _, text in enumerate(texts):
            if text not in self.cached_embeddings:
                need_to_embed.append(text)
        
        # Embed new texts if needed
        if need_to_embed:
            new_embeddings = self.llm_backbone.embedding_model.embed_batch(need_to_embed)
            # Update cache with new embeddings
            for text, embedding in zip(need_to_embed, new_embeddings):
                self.cached_embeddings[text] = embedding
        
        # Create the final ordered list of embeddings
        embeddings = []
        for text in texts:
            embeddings.append(self.cached_embeddings[text])
            
        return embeddings
        


    def has_relation(
        self,
        entity: Knowledge,
        relation: str,
        neighbor: Knowledge
    ) -> bool:
        """
        NEW: We adapted the code to work with the Knowledge Graph Interface of the
        SQA system.
        """
        relations = self._get_relations_of_head_with_cache(entity)
        relations.extend(self._get_relations_of_tail_with_cache(entity))
        
        # logger.debug(f"Checking if {entity.text} has relation {relation} with {neighbor.text}")
        # logger.debug(f"Checking {len(relations)} relations")

        for r in relations:
            if relation in r.predicate:
                if neighbor.uid in r.entity_object.uid or neighbor.uid in r.entity_subject.uid:
                    return True
        return False

    def get_relations_neighbors_set_with_ratings(
        self,
        relations: list,
        neighbors: list,
        query_embedding: list,
    ) -> list:
        """
        given a list of relations and neighbors, return top-n relations and neighbors with the corresponding ratings [(relation, 0.9), (relation, 0.8), ...]
        """
        query_embedding = np.array(query_embedding)

        relations_embeddings = np.array(
            [relation["embedding"] for relation in relations])
        neighbors_embeddings = np.array(
            [neighbor["embedding"] for neighbor in neighbors])

        try:
            # calculate cosine similarity
            query_relation_similarity = self.cos_simiarlity(
                query_embedding, relations_embeddings)
            query_neighbor_similarity = self.cos_simiarlity(
                query_embedding, neighbors_embeddings)

        except Exception:
            logger.warning(
                "Failed to calculate cosine similarity")
            return [], []

        # sort the neighbors by similarity
        relations = [(relations[i], query_relation_similarity[i])
                     for i in np.argsort(query_relation_similarity)[::-1]]

        neighbors = [(neighbors[i], query_neighbor_similarity[i])
                     for i in np.argsort(query_neighbor_similarity)[::-1]]
        # ----> NEW
        # The description of the function mentions that here the Top-N relations
        # should be returned, however in the original code this is not done.
        # Because we encountered enormously long runtimes we added the 
        # restriction.
        relations = relations[:self.args["top_n"]]
        neighbors = neighbors[:self.args["top_n"]]
        # <----

        return relations, neighbors

    def scoring_path(
        self,
        keyword_embeddings: list,
        rated_relations: list,
        rated_neighbors: list,
        hub_node: str,
        reasoning_path: str,
    ) -> list:
        """
        given a list of relations and neighbors with ratings, return top-k relations and neighbors

        NEW: We parallelized this method as it was painfuly slow. We had runtimes above
        10 minutes per question which was unaccetable.
        """
        
        prg_handler = ProgressHandler()
        task = prg_handler.add_task(
            string_id="scoring_paths",
            description="Scoring paths...",
            total=len(rated_relations) * len(rated_neighbors)
        )

        # Create tuples with all combinations of relations and neighbors
        tasks = []
        for relation_with_score in rated_relations:
            for neighbor_with_score in rated_neighbors:
                tasks.append((relation_with_score, neighbor_with_score))

        rated_paths = []  # [(path, score)]c
        seen_paths = set()  # store the seen paths [path, path]
        with ThreadPoolExecutor(max_workers=self.args["max_workers"]) as executor:
            future_to_task = {
                executor.submit(
                    self._process_relation_neighbor_pair,
                    relation_with_score,
                    neighbor_with_score,
                    keyword_embeddings,
                    reasoning_path,
                    hub_node
                ): (relation_with_score, neighbor_with_score)
                for relation_with_score, neighbor_with_score in tasks
            }
            
            for future in as_completed(future_to_task):
                prg_handler.update_task_by_string_id(task, advance=1)
                result = future.result()
                if result is not None:
                    path, score, triple = result
                    if path not in seen_paths:
                        rated_paths.append((path, score, triple))
                        seen_paths.add(path)

        prg_handler.finish_by_string_id(task)
        rated_paths = sorted(rated_paths, key=lambda x: x[1], reverse=True)[
            :self.args["top_n"]]

        # only return the path
        return rated_paths
    
    def _process_relation_neighbor_pair(
        self,
        relation_with_score,
        neighbor_with_score,
        keyword_embeddings,
        reasoning_path,
        hub_node
    ):
        """
        NEW: Helper method to process a single relation-neighbor pair for parallel execution
        """
        relation, relation_score = relation_with_score
        neighbor, neighbor_score = neighbor_with_score
        
        new_rpth = f"{reasoning_path[0]} -> {relation['attribute']} -> {neighbor['attribute'].model_dump_json()}"
        
        # Skip if the relation doesn't exist
        if not self.has_relation(
            entity=hub_node,
            relation=relation['attribute'],
            neighbor=neighbor['attribute']
        ):
            return None
        
        # 1-hop neighbors = relation + neighbor
        one_hop_relations, one_hop_neighbors = self.get_entity_edges(neighbor['attribute'])
        
        if one_hop_relations and one_hop_neighbors:
            one_hop_rated_relations, one_hop_rated_neighbors = self.get_relations_neighbors_set_with_ratings(
                one_hop_relations, one_hop_neighbors, keyword_embeddings)
        else:
            # if there is no one-hop neighbors, set the score to 0
            one_hop_rated_relations, one_hop_rated_neighbors = [(None, 0)], [(None, 0)]
        
        # score function for path_rag
        rpth_score = relation_score + neighbor_score + \
            self.args["alpha"] * (one_hop_rated_relations[0][1] + one_hop_rated_neighbors[0][1])
        
        return (new_rpth, rpth_score, neighbor["triple"])

    def get_path(
        self,
        state: dict
    ) -> list:
        """
        given a starting entity, find top-k one-step path to the query (keywords)

        NEW: We adapted the code to work with the Knowledge Graph Interface of the
        SQA system.
        """
        keywords = state.get(
            "key_words", "")  # using the keywords generated from llm to represent the query
        reasoning_path = state.get("rpth", "")
        
        hub_node = reasoning_path[0].split(" -> ")[-1]
        
        try:
            hub_node = Knowledge.model_validate_json(hub_node)
        except Exception as e:
            logger.error(f"Error creating Knowledge object: {e}")
            raise e

        relations, neighbors = self.get_entity_edges(hub_node)

        embeddings = state.get("embeddings", "")

        if relations and neighbors:
            # get relations and neighbors with the corresponding ratings
            rated_relations, rated_neighbors = self.get_relations_neighbors_set_with_ratings(
                relations, neighbors, embeddings)

        else:
            return []

        # top-n scoring paths
        paths = self.scoring_path(
            keyword_embeddings=embeddings,
            reasoning_path=reasoning_path,
            rated_relations=rated_relations,
            rated_neighbors=rated_neighbors,
            hub_node=hub_node,)

        return paths

    def _get_relations_of_head_with_cache(self, entity: Knowledge):
        if entity.uid not in self.cached_entity_relations_head:
            relations = self.graph.get_relations_of_head_entity(entity)
            if not relations:
                return []
            self.cached_entity_relations_head[entity.uid] = relations.copy()
        else:
            relations = self.cached_entity_relations_head[entity.uid]

        return relations
    
    def _get_relations_of_tail_with_cache(self, entity: Knowledge):
        if entity.uid not in self.cached_entity_relations_tail:
            relations = self.graph.get_relations_of_tail_entity(entity)
            if not relations:
                return []
            self.cached_entity_relations_tail[entity.uid] = relations.copy()
        else:
            relations = self.cached_entity_relations_tail[entity.uid]
        return relations