import pickle
from typing import Dict, List, Tuple
import ast
import re
from collections import defaultdict

from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.models import Triple
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

from .struct_gpt_chatgpt import ChatGPT
from .struct_gpt_retriever import Retriever

logger = get_logger(__name__)

class Solver:
    def __init__(self, args, graph: KnowledgeGraph) -> None:
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name)
        self.SLM = Retriever(args, graph)
        self.graph = graph
        self.max_serialization_tokens = args.max_llm_input_tokens
        # self.load_ent2name(args.ent2name_path)
        self.log = []
        self.selected_relations = []
        # 暂时添加一个selected_sub_questions = []来存放解析的子问题
        self.selected_sub_questions = []
        # ----> NEW
        # Added new attributes to the class
        self.bidirectional = getattr(args, "bidirectional", False)
        self.replace_contribution = getattr(args, "replace_contribution_name", True)
        self.progress_handler = ProgressHandler()
        # <----

    def forward_v2(self, question, tpe_str, tpe_id):
        self.LLM.reset_history()
        self.SLM.reset_cur_ents([tpe_id])
        self.reset_history()

        iterative_step = 0
        # ----> NEW
        # Here the original code would not stop if the max_iterations was reached.
        # As the while loop was set to "while TRUE". Therefore we adapted the code
        # to check if the max_iterations was reached.
        max_iterations = self.args.max_depth
        final_filtered_triples = {}
        progress_task = self.progress_handler.add_task(
            description="Searching Depth",
            string_id="searching_depth",
            total=max_iterations,
            reset=True
        )
        while iterative_step < max_iterations:
        # <----
            # select
            all_rel_one_hop: List[Triple] = self.SLM.get_retrieval_relations(iterative_step,
                                                               bidirectional=self.bidirectional)
            if len(all_rel_one_hop) == 0:
                final_answers = self.LLM.get_response_v2(question, "final_query_template")
                break

            # serialized_rels = self.extract_can_rels(all_rel_one_hop, normalize_rel=False)
            # ----> NEW
            # We found the LLM to have difficulties if the predicate is "contribution" and work
            # better with an alternative string such as "paper content". We added this functionality
            # for testing purposes.
            predicates = [rel.predicate for rel in all_rel_one_hop]
            predicates = list(set(predicates))
            if self.replace_contribution:
                predicates = self._replace_string(predicates, "contribution", "paper content")            
            serialized_rels = ", ".join(predicates)
            
            
            logger.debug("Step-%d: serialized_rels:%s" % (iterative_step, serialized_rels))
            # <----
            self.LLM.reset_history_messages()
                
            if iterative_step == 0:
                llm_selected_rels = self.LLM.get_response_v2((serialized_rels, question, tpe_str),
                                                             "init_relation_rerank")
            else:
                llm_selected_rels = self.LLM.get_response_v2(
                    (serialized_rels, question, tpe_str, self.selected_relations), "relation_rerank")
            self.LLM.reset_history_messages()
            
            logger.debug("Step-%d: llm_selected_rels:%s" % (iterative_step, llm_selected_rels))

            selected_relations_list = self.parse_llm_selected_relations(llm_selected_rels, predicates)
            # ----> NEW
            # We found the LLM to have difficulties if the predicate is "contribution" and work
            # better with an alternative string such as "paper content". We added this functionality
            # for testing purposes.
            if self.replace_contribution:
                selected_relations_list = self._replace_string(selected_relations_list, "paper content", "contribution")
            # <----
            # ----> NEW
            # Added logging
            logger.debug("Step-%d: selected_relations_list:%s" % (iterative_step, selected_relations_list))
            # <----
            if len(selected_relations_list) == 0:
                final_answers = self.LLM.get_response_v2(question, "final_query_template")
                break

            self.selected_relations.extend(selected_relations_list)
            # ----> NEW
            # Added logging
            logger.debug("Step-%d: self.selected_relations:%s" % (iterative_step, self.selected_relations))
            # <----

            filtered_triples_per_hop, _ = self.SLM.get_retrieval_information(iterative_step,
                                                                              gold_relations=selected_relations_list,
                                                                              all_rel_one_hop=all_rel_one_hop,
                                                                              bidirectional=self.bidirectional)
            # ----> NEW
            # We can skip this because we are using a different graph which does not have these specific classifications
            # cvt_triples, mid_triples, entstr_triples = self.classify_triples(filtered_triples_per_hop)
            # if len(cvt_triples) > 0:
            #     # constraint
                
            #     logger.debug("Step-%d: Constraints" % iterative_step)
            #     constraints_candidate = self.serialize_constraints(cvt_triples)
                
            #     logger.debug("Step-%d: constraints_candidate:%s" % (iterative_step, constraints_candidate))
            #     constraint_response = self.LLM.get_response_v2((question, constraints_candidate, tpe_str), "choose_constraints")
            #     self.log.append(constraint_response)
                
            #     logger.debug("Step-%d: constraint_response:%s" % (iterative_step, constraint_response))
            #     if self.has_constraints(constraint_response):
            #         filtered_triples_per_hop = self.filter_triples(filtered_triples_per_hop, cvt_triples,
            #                                                        constraint_response)
            #         self.SLM.update_cur_ents(filtered_triples_per_hop)
                    
            #         logger.debug("Step-%d: filtered_triples_per_hop:%s" % (iterative_step, filtered_triples_per_hop))
                    
            #         logger.debug("Step-%d: self.SLM.cur_ents:%s" % (iterative_step, self.SLM.cur_ents))
            # <----
            
            serialized_facts, triples_that_were_serialized = self.serialize_facts(filtered_triples_per_hop)
            
            final_filtered_triples.update({iterative_step: triples_that_were_serialized})
            self.log.append(serialized_facts)
            
            logger.debug("Step-%d: serialized_facts:%s" % (iterative_step, serialized_facts))

            final_ans_or_next_que = self.LLM.get_response_v2((question, serialized_facts),
                                                             "ask_final_answer_or_next_question")
            self.log.append(final_ans_or_next_que)

            iterative_step += 1
            # ----> NEW
            # Here the original code has an unconditional break statement which 
            # doesn't allow the retrieval to go beyond 1-hop.
            # Therefore we implemented a check to see if we have a final answer or need to continue.
            # Check if we have a final answer or need to continue
            if self.is_final_answer(final_ans_or_next_que) or iterative_step >= max_iterations:
                final_answers = self.parse_result(final_ans_or_next_que, "final_answer")
                logger.debug(f"Final answer has been found: {final_answers}")
                self.log.append(final_answers)
                break
            logger.debug("Have not yet found a final answer. Doing next Iteration")
            self.progress_handler.update_task_by_string_id(progress_task)
        self.progress_handler.finish_by_string_id(progress_task)
        filtered_triples = self.filter_relevant_triples(final_filtered_triples, final_answers)
        return final_answers, self.LLM.history_contents, self.log, filtered_triples
        # <----
    
    def filter_relevant_triples(self, filtered_triples_per_hop: dict[str, List[Triple]], final_ans: str):
        """
        NEW: We added a filtering function to filter the relevant triples based on the final answer.
        Filters the relevant triples from the filtered triples per hop based on the final answer.
        
        Args:
            filtered_triples_per_hop: Filtered triples per hop
            final_ans: Final answer
        """
        
        if len(filtered_triples_per_hop) == 0:
            return []
        
        formated_triples = []
        # Get the last hop 
        last_hop = max(filtered_triples_per_hop.keys())
        # Get the filtered triples for the last hop
        for index, triple in enumerate(filtered_triples_per_hop[last_hop]):
            formated_triples.append(f"{index}: {triple.entity_subject.text} {triple.predicate} {triple.entity_object.text}")
        
        prompt = """
            Given an answer to a question, select all the context that directly provides the information for the answer. The context is in the form of 'ID: (subject, relation, object)'. You have to return a list of IDs that are relevant to the answer e.g. [2, 5, 9].
            The answer: The evaluation method is named Interview.
            The contexts: 0: (Paper, has title, Microservice Architecture for Cloud Computing), 1: (Paper, has author, John Doe), 2: (Paper, has evaluation method, Interview), 3: (Paper, has evaluation method, Survey)
            Output: [2, 3]
            The answer: The authors are John Doe and Jane Smith.
            The contexts: 0: (Paper, has title, Microservice Architecture for Cloud Computing), 1: (Paper, has author, John Doe), 2: (Paper, has evaluation method, Interview), 3: (Paper, has evaluation method, Survey), 4: (Paper, has author, Jane Smith)
            Output: [1, 4]
            The answer: {final_ans} \n\n
            The contexts: {formated_triples}
            Output: 
        """
        formatted_prompt = prompt.format(final_ans=final_ans, formated_triples='\n'.join(formated_triples))
        logger.debug(f"Prompt for filtering relevant triples: {formatted_prompt}")
        result = self.LLM.llm_adapter.generate(formatted_prompt)
        logger.debug(f"LLM output for filtering relevant triples: {result.content}")
        extracted_ids = self._extract_id_list(result.content)
        logger.debug(f"Extracted IDs from LLM output: {extracted_ids}")
                
        # Filter the relevant triples based on the extracted IDs
        # Check if the extracted ids are valid
        if all(id < len(formated_triples) for id in extracted_ids):
            relevant_triples = [filtered_triples_per_hop[last_hop][id] for id in extracted_ids]
            logger.debug(f"Final retrieved Contexts: {relevant_triples}")
            return relevant_triples
        
        logger.warning("The extracted IDs are not valid.")
        return filtered_triples_per_hop[last_hop]
        
    def _extract_id_list(self, llm_output: str) -> List[int]:
        list_match = re.search(r'\[[^\]]*\]', llm_output)

        if list_match:
            list_str = list_match.group(0)
            try:
                # Attempt to safely evaluate the extracted string as a Python literal
                potential_list = ast.literal_eval(list_str)
                if isinstance(potential_list, list) and all(isinstance(item, int) for item in potential_list):
                    return potential_list
            except (ValueError, SyntaxError):
                pass

            # Fallback: manually extract all integers within the brackets
            return [int(num) for num in re.findall(r'\d+', list_str)]

        # If no brackets found, check if the entire output is just numbers separated by commas
        stripped_text = llm_output.strip()
        if re.fullmatch(r'(\d+\s*,\s*)*\d+', stripped_text):
            # The entire output is a comma-separated list of numbers
            return [int(num.strip()) for num in stripped_text.split(',')]

        # Fallback: find all integers anywhere in the text
        return [int(num) for num in re.findall(r'\d+', llm_output)]
    
    def _replace_string(self, str_list: List[str], replace:str, replace_with: str) -> List[str]:
        """
        Replaces the string in str_list with replace_with.
        
        Args:
            str_list: List of strings
            replace: String to be replaced
            replace_with: String to replace with
        """
        for i, s in enumerate(str_list):
            if replace in s:
                str_list[i] = s.replace(replace, replace_with)
        return str_list
    
    def is_final_answer(self, final_answers):
        if "final answer" in final_answers.lower():
            return True
        else:
            return False

    def reset_selected_list(self):
        self.selected_sub_questions = []
        self.selected_relations = []

    def is_end(self, response, iterative_step):
        if "no" in response.lower() or iterative_step > 8:
            return True
        else:
            return False

    def load_ent2name(self, ent2name_path):
        with open(ent2name_path, "rb") as f:
            self.cvt_flag_dict, self.mid_mapping_dict = pickle.load(f)

    def convert_hyper_facts_to_text(self, facts):
        subj, rels, objs = facts

        if self.is_cvt(subj):
            return None
        elif subj in self.mid_mapping_dict:
            subj_surface = self.mid_mapping_dict[subj]
        elif self.is_ent(subj):
            # print("head entity %s doesn't have name, we skip this triple." % subj)
            return None
        else:
            subj_surface = subj

        flat_facts = []
        for rel, obj in zip(rels, objs):
            if self.should_ignore(rel):
                continue
            else:
                nor_rel = self.normalize_relation(rel)

            if self.is_cvt(obj):
                continue
            elif obj in self.mid_mapping_dict:
                obj_surface = self.mid_mapping_dict[obj]
            elif self.is_ent(obj):
                # print("tail entity %s doesn't have name, we skip this triple." % obj)
                continue
            else:
                obj_surface = obj

            flat_facts.append((subj_surface, nor_rel, obj_surface))

        return flat_facts

    def convert_fact_to_text(self, fact, normalize_rel=False):
        subj, rel, obj = fact

        if self.should_ignore(rel):
            return None

        if rel.endswith(".from"):
            rel = rel.rstrip(".from")
            rel = rel + ".start_time"
        if rel.endswith(".to"):
            rel = rel.rstrip(".to")
            rel = rel + ".end_time"
        rel_surface = self.normalize_relation(rel) if normalize_rel else rel

        # subject
        if subj.startswith("CVT"):
            subj_surface = subj
        elif subj in self.mid_mapping_dict:
            subj_surface = self.mid_mapping_dict[subj]
        elif subj.startswith("m.") or subj.startswith('g.'):
            # print("head entity %s doesn't have name, we skip this triple." % subj)
            return None
        else:
            subj_surface = subj

        # object
        if obj.startswith("CVT"):
            obj_surface = obj
        elif obj in self.mid_mapping_dict:
            obj_surface = self.mid_mapping_dict[obj]
        elif obj.startswith("m.") or obj.startswith('g.'):
            # print("tail entity %s doesn't have name, we skip this triple." % obj)
            return None
        else:
            obj_surface = obj

        return (subj_surface, rel_surface, obj_surface)

    def extract_can_rels(self, all_rel_one_hop, normalize_rel=True):
        rel_prompt = '"{relation}"'
        nor_rels_set = []
        for rel in all_rel_one_hop:
            nor_r = self.normalize_relation(rel) if normalize_rel else rel
            if nor_r not in nor_rels_set:
                nor_rels_set.append(rel_prompt.format(relation=nor_r)) 
        rel_candidate = ", ".join(all_rel_one_hop)
        return rel_candidate

    def serialize_rels(self, rels, normalize_rel=True):
        nor_rels_set = []
        for rel in rels:
            if self.filter_relation(rel):
                continue
            nor_r = self.normalize_relation(rel) if normalize_rel else rel
            if nor_r not in nor_rels_set:
                nor_rels_set.append(nor_r)
        # rel_candidate = ", ".join(nor_rels_set)
        rel_candidate = ";\n ".join(nor_rels_set)
        return rel_candidate

    # 直接拼接
    def serialize_facts_direct(self, facts):
        # 拼接triples
        facts_str_for_one_tail_ent = ["(" + ", ".join(fact) + ")" for fact in facts]

        serialized_facts = ""
        for fact in facts_str_for_one_tail_ent:
            serialized_facts_tmp = serialized_facts + fact + "; "
            serialized_facts = serialized_facts_tmp
        return serialized_facts

    def serialize_facts(self, facts_per_hop: Dict[int, List[Triple]]) -> str:
        """
        NEW: Updated function to work with the Knowledge Graph interface of the
        SQA System.
        """
        all_facts: List[Triple] = []
        for hop, facts in facts_per_hop.items():
            all_facts.extend(facts)

        if not all_facts:
            return ""

        serialized_triples = []
        serialized_facts = []
        token_count = 0
        for triple in all_facts:
            head_name = triple.entity_subject.text
            tail_name = triple.entity_object.text

            fact_str = f"({head_name}, {triple.predicate}, {tail_name})"
            serialized_facts.append(fact_str)
            serialized_triples.append(triple)
            
            token_count += len(fact_str.split())
            if token_count > self.max_serialization_tokens:
                break

        # Join all facts and limit to max_serialization_tokens
        serialized_facts_str = "; ".join(serialized_facts)

        return serialized_facts_str, serialized_triples

    def serialize_facts_v1(self, facts):
        if len(facts) > 0:
            h_r_t = defaultdict(lambda: defaultdict(set))
            visited_flag = {}
            for fact in facts:
                h, r, t = fact
                visited_flag[tuple(fact)] = False
                h_r_t[h][r].add(t)
            facts_str = []
            for tri in facts:
                if not visited_flag[tuple(tri)]:
                    h, r, t = tri
                    if self.is_cvt(t) and len(h_r_t[t]) == 0:
                        continue
                    if self.is_cvt(h):
                        # print("Qid:[%s] has single cvt head entities." % qid)
                        # logger.info(triples_per_hop)
                        continue
                    elif self.is_cvt(t):
                        one_hop_triples = h_r_t[t]
                        if len(one_hop_triples) > 0:
                            h_new = t
                            r_new = []
                            t_new = []
                            for key_r, value_ts in one_hop_triples.items():
                                for t_ in value_ts:
                                    visit_tri = (t, key_r, t_)
                                    if not visited_flag[visit_tri]:
                                        r_new.append(key_r)
                                        t_new.append(t_)
                                        visited_flag[visit_tri] = True
                            tri_new = (h, r_new, t_new)
                            if len(r_new) == len(t_new) > 0:
                                str_tri_list = self.convert_hyper_facts_to_text(tri_new)
                                if str_tri_list is not None:
                                    for st in str_tri_list:
                                        assert len(st) == 3
                                        if st not in facts_str:
                                            facts_str.append(st)
                    else:
                        st = self.convert_fact_to_text(tri)
                        if st is not None:
                            assert len(st) == 3
                            if st not in facts_str:
                                facts_str.append(st)
            facts_str = ["(" + ", ".join(fact) + ")" for fact in facts_str]
            serialized_facts = ""
            for fact in facts_str:
                serialized_facts_tmp = serialized_facts + fact + "; "
                if len(serialized_facts_tmp.split()) > self.max_serialization_tokens:
                    break
                else:
                    serialized_facts = serialized_facts_tmp
            # serialized_facts = "; ".join(facts_str)
            serialized_facts = serialized_facts.strip("; ")
        else:
            serialized_facts = ""
        return serialized_facts

    def is_cvt(self, entity):
        if self.cvt_flag_dict[entity]:
            return True
        else:
            return False

    def is_ent(self, ent_str):
        return self.graph.is_intermediate_id(ent_str)
            
        # if type(ent_str) is not bool and (ent_str.startswith("m.") or ent_str.startswith("g.")):
        #     return True
        # else:
        #     return False

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

    def parse_llm_selected_relations(self, llm_sel_rels_str, can_rels):
        # llm_sel_rels = llm_sel_rels_str.strip(" ;.|,<>`[]'")
        # llm_sel_rels = llm_sel_rels.split(',')
        # llm_sel_rels = [rel.strip(" ;.|,<>`[]'").strip(" ;.|,<>`[]'") for rel in llm_sel_rels]
        # llm_sel_rel_list = []
        # for rel in llm_sel_rels:
        #     if rel in can_rels:
        #         llm_sel_rel_list.append(rel)
        #     else:
        #         print(rel)
        # if len(llm_sel_rel_list) == 0:
        #     for rel in can_rels:
        #         if rel in llm_sel_rels_str:
        #             llm_sel_rel_list.append(rel)
        #     print("-----llm_ser_rels:\n%s\ndoesn't match the predefined format" % llm_sel_rels)
        llm_sel_rel_list = []
        for rel in can_rels:
            if rel.lower() in llm_sel_rels_str.lower():
                llm_sel_rel_list.append(rel)
        return llm_sel_rel_list

    def parse_result(self, response, parse_type):
        response = response.lower()
        if parse_type == "next_question":
            if "the next question:" in response:
                next_question = response.split("the next question:")[1].strip()
            elif ":" in response:
                next_question = response.split(":")[1].strip()
            else:
                next_question = response
            return next_question
        elif parse_type == "final_answer":
            if "the final answers:" in response:
                final_answer = response.split("the final answers:")[1].strip()
            # 暂时注释掉
            elif ":" in response:
                final_answer = response.split(":")[1].strip()
            # 新添加的用于解析direct query
            else:
                final_answer = response
                # 暂时注释掉
                # print("Not parse the final answer exactly, directly use the response: ", response)
            return final_answer

    def classify_triples(self, filtered_triples_per_hop):
        cvt_triples, mid_triples, entstr_triples = set(), set(), set()
        if 0 in filtered_triples_per_hop:
            triples_0 = filtered_triples_per_hop[0]
        else:
            triples_0 = []
        if 1 in filtered_triples_per_hop:
            triples_1 = filtered_triples_per_hop[1]
        else:
            triples_1 = []

        if len(triples_1) == 0:
            for tri in triples_0:
                if self.is_ent(tri[2]):
                    mid_triples.add(tuple(tri))
                else:
                    entstr_triples.add(tuple(tri))
        else:
            for tri in triples_1:
                cvt_triples.add(tuple(tri))
        return cvt_triples, mid_triples, entstr_triples

    def serialize_constraints(self, cvt_triples):
        r2t_set = defaultdict(set)
        for tri in cvt_triples:
            subj, rel, obj = tri
            if self.should_ignore(rel):
                continue

            if rel.endswith(".from"):
                rel = rel.rstrip(".from")
                rel = rel + ".start_time"
            if rel.endswith(".to"):
                rel = rel.rstrip(".to")
                rel = rel + ".end_time"

            rel_surface = rel

            # object
            if obj in self.mid_mapping_dict:
                obj_surface = self.mid_mapping_dict[obj]
            elif obj.startswith("m.") or obj.startswith('g.'):
                # print("tail entity %s doesn't have name, we skip this triple." % obj)
                continue
            else:
                obj_surface = obj

            if obj_surface == "To" or "has_no_value" in rel:
                continue

            r2t_set[rel_surface].add(obj_surface)

        constraints = []
        for r, t_set in r2t_set.items():
            t_set = ['"' + t + '"' for t in t_set]
            constraints.append('"' + r + '"' + ": [" + ", ".join(t_set) + "]")
        # constraints = constraints.rstrip("\n")
        constraints = "\n".join(constraints)
        return constraints

    def has_constraints(self, constraint_response):
        if "no" in constraint_response.lower():
            return False
        else:
            return True

    def filter_triples(self, filtered_triples_per_hop, cvt_triples, constraint_response):
        valid_cvt_nodes = set()
        h_r_t = defaultdict(list)
        for tri in cvt_triples:
            h, r, t = tri
            h_r_t[h].append((r, t))
        for cvt, r_ts in h_r_t.items():
            flag = True
            at_leat_one_flag = False
            for r_t in r_ts:
                rel, obj = r_t
                if rel.endswith(".from"):
                    rel = rel.rstrip(".from")
                    rel = rel + ".start_time"
                if rel.endswith(".to"):
                    rel = rel.rstrip(".to")
                    rel = rel + ".end_time"
                rel_surface = rel

                if obj in self.mid_mapping_dict:
                    obj_surface = self.mid_mapping_dict[obj]
                elif obj.startswith("m.") or obj.startswith('g.'):
                    # print("tail entity %s doesn't have name, we skip this triple." % obj)
                    continue
                else:
                    obj_surface = obj

                if rel_surface.lower() in constraint_response.lower():
                    at_leat_one_flag = True
                    if obj_surface.lower() not in constraint_response.lower():
                        flag = False
                        break
            if flag and at_leat_one_flag:
                valid_cvt_nodes.add(cvt)

        if len(valid_cvt_nodes) == 0:
            for cvt, r_ts in h_r_t.items():
                flag = True
                at_leat_one_flag = False
                for r_t in r_ts:
                    rel, obj = r_t

                    if rel.endswith(".from"):
                        rel = rel.rstrip(".from")
                        rel = rel + ".start_time"
                    if rel.endswith(".to"):
                        rel = rel.rstrip(".to")
                        rel = rel + ".end_time"
                    rel_surface = rel

                    if obj in self.mid_mapping_dict:
                        obj_surface = self.mid_mapping_dict[obj]
                    elif obj.startswith("m.") or obj.startswith('g.'):
                        # print("tail entity %s doesn't have name, we skip this triple." % obj)
                        continue
                    else:
                        obj_surface = obj

                    rel_surface_list = rel_surface.split(".")
                    for rel in rel_surface_list:
                        if rel.lower() in constraint_response.lower():
                            at_leat_one_flag = True
                            if obj_surface.lower() not in constraint_response.lower():
                                flag = False
                                break
                            else:
                                flag = True
                    if flag and at_leat_one_flag:
                        valid_cvt_nodes.add(cvt)
                        break

        new_tris_per_hop = defaultdict(set)
        for hop in [0, 1]:
            triples = filtered_triples_per_hop[hop]
            for tri in triples:
                h, r, t = tri
                if hop == 0:
                    if t in valid_cvt_nodes:
                        new_tris_per_hop[hop].add(tuple(tri))
                elif hop == 1:
                    if h in valid_cvt_nodes:
                        new_tris_per_hop[hop].add(tuple(tri))
        return new_tris_per_hop

    def serialize_facts_one_hop(self, facts):
        if len(facts) > 0:
            h_r_t = defaultdict(lambda: defaultdict(set))
            visited_flag = {}
            for fact in facts:
                h, r, t = fact
                visited_flag[tuple(fact)] = False
                h_r_t[h][r].add(t)
            facts_str = []
            for tri in facts:
                if not visited_flag[tuple(tri)]:
                    h, r, t = tri
                    if self.is_cvt(t) and len(h_r_t[t]) == 0:
                        continue
                    if self.is_cvt(h):
                        # print("Qid:[%s] has single cvt head entities." % qid)
                        # logger.info(triples_per_hop)
                        continue
                    elif self.is_cvt(t):
                        one_hop_triples = h_r_t[t]
                        if len(one_hop_triples) > 0:
                            h_new = t
                            r_new = []
                            t_new = []
                            for key_r, value_ts in one_hop_triples.items():
                                for t_ in value_ts:
                                    visit_tri = (t, key_r, t_)
                                    if not visited_flag[visit_tri]:
                                        r_new.append(key_r)
                                        t_new.append(t_)
                                        visited_flag[visit_tri] = True
                            tri_new = (h, r_new, t_new)
                            if len(r_new) == len(t_new) > 0:
                                str_tri_list = self.convert_hyper_facts_to_text(tri_new)
                                if str_tri_list is not None:
                                    for st in str_tri_list:
                                        assert len(st) == 3
                                        if st not in facts_str:
                                            facts_str.append(st)
                    else:
                        st = self.convert_fact_to_text(tri)
                        if st is not None:
                            assert len(st) == 3
                            if st not in facts_str:
                                facts_str.append(st)
            facts_str = ["(" + ", ".join(fact) + ")" for fact in facts_str]
            serialized_facts = ""
            for fact in facts_str:
                serialized_facts_tmp = serialized_facts + fact + "; "
                if len(serialized_facts_tmp.split()) > self.max_serialization_tokens:
                    break
                else:
                    serialized_facts = serialized_facts_tmp
            # serialized_facts = "; ".join(facts_str)
            serialized_facts = serialized_facts.strip("; ")
        else:
            serialized_facts = ""
        return serialized_facts

    def is_end_v2(self, response, iterative_step):
        if "final" in response.lower() or iterative_step > 3:
            return True
        else:
            return False

    def reset_history(self):
        self.log = []
        self.selected_relations = []
        self.selected_sub_questions = []

    def get_tails_list(self, cur_ents):
        tails = self.SLM.get_tails_list(cur_ents)
        return tails