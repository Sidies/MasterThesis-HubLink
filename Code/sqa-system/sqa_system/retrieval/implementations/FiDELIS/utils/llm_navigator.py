import datetime
from typing import Optional
import copy
import time
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqa_system.core.logging.logging import get_logger
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.data.models import Knowledge

from .llm_backbone import LLM_Backbone
from .path_rag import Path_RAG
from ..prompts import fidelis_prompts

logger = get_logger(__name__)


class LLMNavigator():

    def __init__(self, graph: KnowledgeGraph, args: dict):
        self.args = args
        self.graph = graph
        self.path_rag_engine = Path_RAG(graph, args)
        self.llm_backbone = LLM_Backbone(
            llm_config=self.args["llm_config"],
            embedding_config=self.args["embedding_config"]
        )
        self.prompt_list = fidelis_prompts
        self._new_line_char = "\n"  # for formatting the prompt

    def beam_search(self, query_text: str, topic_entity_id: Optional[str]):
        """
        NEW: We adapted the code to work with the Knowledge Graph Interface of the
        SQA system.
        """

        # Get the starting node
        start_node = self.graph.get_entity_by_id(topic_entity_id)
        if not start_node:
            logger.warning(f"Could not find the topic entity with ID: {topic_entity_id}")
            return None

        llm_states = {}
        llm_states["entity"] = start_node
        llm_states["question"] = query_text
        llm_states["starting_entities"] = [start_node]
        self.planning(llm_states)        

        # Initialize the progress handler
        progress_handler = ProgressHandler()
        task = progress_handler.add_task(
            string_id="beam_search",
            description="Beam searching...",
            total=self.args["max_length"],
        )

        reasoning_paths = []  # final reasoning paths
        # store the reasoning paths for each step, the the length of the list is equal to the number of top-k
        active_beam_reasoning_paths = [[(str(start_node.model_dump_json()), None, None)]]
        pred_list_direct_answer = []
        pred_list_llm_reasoning = []
        reasoning_path_list = []
        for step in range(self.args["max_length"]):
            all_candidates = []
            
            # The active beam search would not stop until all paths have been explored
            # which leads to long runtimes. Here we prematurely stop if reasoning paths
            # are found
            if self.args["prematurely_stop_when_paths_are_found"] and reasoning_paths:
                break
            task_2 = progress_handler.add_task(
                string_id="beam_search_step",
                description="Traversing Paths...",
                total=len(active_beam_reasoning_paths),
                reset=True,
            )

            with ThreadPoolExecutor(max_workers=self.args["max_workers"]) as executor:
                futures = {}
                for rpth in active_beam_reasoning_paths:
                    future = executor.submit(
                        self._process_path_helper, rpth, llm_states, step)
                    futures[future] = rpth
                    
                for future in as_completed(futures):
                    next_step_candidates, found_answer = future.result()
                    if found_answer:
                        reasoning_paths.append(futures[future])
                        continue
                    if next_step_candidates:  # if there are no next_step_candidates, skip the current step
                        all_candidates.extend(next_step_candidates)
                    progress_handler.update_task_by_string_id(task_2)

            if not all_candidates:
                break

            if step != self.args["max_length"]:  # if not the last step
                llm_states["next_step_candidates"] = all_candidates
                active_beam_reasoning_paths = self.decide_top_k_candidates(
                    state=llm_states
                )

                logger.debug("<<<<<<<<")
                logger.debug("Active Beam Reasoning Paths: {}".format(
                    active_beam_reasoning_paths))
                logger.debug(">>>>>>>>")
            progress_handler.update_task_by_string_id(task, advance=1)
        progress_handler.finish_by_string_id(task)

        # if there are no candidates fit the criteria, return the active_beam_raesoning_paths
        if not reasoning_paths:
            reasoning_paths = active_beam_reasoning_paths

        llm_states["reasoning_paths"] = reasoning_paths

        # --------------
        # LLM REASONING
        # --------------
        reasoning_res = self.reasoning(llm_states)

        for item in reasoning_res.split(", "):
            pred_list_llm_reasoning.append(item)

        for item in reasoning_paths:
            pred_list_direct_answer.append(item[0][2])
            reasoning_path_list.append(item[0])

        # save the results to a dict 
        res = {
            "question": query_text,
            "q_entities": [start_node],
            "reasoning_path": reasoning_path_list,
            # remove duplicate predictions
            "prediction_llm": "\n".join(set(pred_list_llm_reasoning)),
            "prediction_direct_answer": pred_list_direct_answer
        }
        return res
    
    def _process_path_helper(self, rpth, llm_states, step):
        """
        NEW: Helper function to process the path in parallel.
        """
        updated_llm_states = llm_states.copy()
        updated_llm_states["rpth"] = rpth[0]
        reasoning_paths = []
        # if meet the condition, skip the current step
        if step != 0:
            flag = self.deductive_termination(
                state=updated_llm_states
            )
            if flag:
                # The question can be answered with the path
                reasoning_paths.append(rpth)
                return reasoning_paths, True

        next_step_candidates = self.path_rag_engine.get_path(
            state=updated_llm_states
        )
        return next_step_candidates, False

    def rpth_parser(
        self,
        state: dict
    ):
        """
        Reformulate the reasoning path from the agent state
        """
        reasoning_path = state.get("rpth", "")
        formatted_reasoning_path = self._remove_knowledge_formatting_from_reasoning_path(reasoning_path[0])
        reformulate_prompt = copy.copy(
            self.prompt_list.reasoning_path_parser_prompt)
        reformulate_prompt["prompt"] = reformulate_prompt["prompt"].format(
            reasoning_path=formatted_reasoning_path
        )
        reformulate_res = self.llm_backbone.get_completion(reformulate_prompt)
        state["parsed_rpth"] = reformulate_res

    def deductive_termination(
        self,
        state: dict
    ):
        """
        NEW: We adapted the code to work with either deductive reasoning or
        terminals pruning. Furthermore, we ensure that the prompt is formatted
        correctly by removing knowledge graph formatting.
        """
        self.rpth_parser(state)  # reformulate the reasoning path

        reasoning_path = state.get("rpth", "")
        reasoning_path = self._remove_knowledge_formatting_from_reasoning_path(reasoning_path[0])
        parsed_reasoning_path = state.get("parsed_rpth", "")
        question = state.get("question", "")
        planning_steps = state.get("planning_steps", "")
        declarative_statement = state.get("declarative_statement", "")

        placeholder_entity = reasoning_path.split(" -> ")[-1]
        declarative_statement = declarative_statement.replace(
            "*placeholder*", placeholder_entity).strip(".")

        if self.args["use_deductive_reasoning"]:
            condition_prompt = copy.copy(
                self.prompt_list.deductive_verifier_prompt)
            condition_prompt["prompt"] = condition_prompt["prompt"].format(
                parsed_reasoning_path=parsed_reasoning_path,
                declarative_statement=declarative_statement
            )
        else:
            condition_prompt = copy.copy(self.prompt_list.terminals_prune_single_prompt)
            condition_prompt["prompt"] = condition_prompt["prompt"].format(
                question=question,
                reasoning_path=reasoning_path,
                plan_context=planning_steps,
            )

        res = self.llm_backbone.get_completion(
                condition_prompt).replace("Answer: ", "").strip()
        # print("Condition Prompt: ", condition_prompt["prompt"], "Deductive Termination: ", res)

        logger.debug("<<<<<<<<")
        logger.debug("Deductive Termination Prompt: {}".format(
            condition_prompt["prompt"]))
        logger.debug("Prediction: {}".format(res))
        logger.debug(">>>>>>>>")

        if "Yes".lower() in res.lower():
            return True
        elif "No".lower() in res.lower():
            return False
        else:
            return False

    def decide_top_k_candidates(
        self,
        state: dict
    ):

        next_step_candidates = state.get("next_step_candidates", [])
        # ----> NEW
        # Remove the knowledge graph formatting from the reasoning path
        reduced_next_step_candidates = [path[0] for path in next_step_candidates]
        reduced_formatted_candidates = [
            self._remove_knowledge_formatting_from_reasoning_path(path) for path in reduced_next_step_candidates
        ]
        # <----
        question = state.get("question", "")
        planning_steps = state.get("planning_steps", "")

        formatted_next_step_candidates = [
            f"{i+1}: {item}" for i, item in enumerate(reduced_formatted_candidates)]
        rating_prompt = copy.copy(self.prompt_list.beam_search_prompt)
        rating_prompt["prompt"] = self.prompt_list.beam_search_prompt["prompt"].format(
            beam_width=self.args["top_k"],
            plan_context=planning_steps,
            question=question,
            reasoning_paths=self._new_line_char.join(
                formatted_next_step_candidates)
        )

        logger.debug("<<<<<<<<")
        logger.debug("Beam Search Prompt: {}".format(rating_prompt["prompt"]))
        logger.debug(">>>>>>>>")

        attempt = 0
        while attempt < 5:  # try 5 times if the index is not found or not as expected
            try:
                rating_index = self.llm_backbone.get_completion(rating_prompt)
                matched_indices = self.extract_indices(
                    rating_index, len(next_step_candidates))

                logger.debug("<<<<<<<<")
                logger.debug("Top-k Indices: {}".format(matched_indices))
                logger.debug(">>>>>>>>")

                top_k_candidates = [[next_step_candidates[i]] for i in matched_indices]
                return top_k_candidates

            except Exception as e:
                logger.error(f"Error occurred: {e}")
                # ----> NEW
                # If the LLM fails to provide a valid response, we add the error
                # message to the prompt and retry
                error_text = str(e).replace("{", "").replace("}", "")
                rating_prompt["prompt"] = (rating_prompt["prompt"] + 
                                           f"\nIn the last call you made an error. Make sure to correct it this time. Error: {error_text}")
                # <----
                attempt += 1
                time.sleep(1)
                
    def extract_indices(self, rating_index: str, num_candidates: int) -> list:
        """
        NEW: Added a new parser to extract indizes from the outputs of the LLM
        as we found that often times the extraction would fail.
        This parser should be more robust.
        """
        
        rating_index = rating_index.replace("Answer:", "").strip()

        numbers = re.findall(r'\d+', rating_index)
        
        if not numbers:
            raise ValueError("No valid number found in the response.")

        candidate_indices = [int(num) - 1 for num in numbers]
        valid_indices = [i for i in candidate_indices if 0 <= i < num_candidates]

        if not valid_indices:
            raise ValueError("Extracted indices are out of the range of available candidates.")

        return valid_indices

    def reasoning(
        self,
        state: dict
    ):
        reasoning_paths = state.get("reasoning_paths", [])
        # ----> NEW
        # Remove the knowledge graph formatting from the reasoning path
        formatted_reasoning_path = ""
        for item in reasoning_paths:
            path = item[0]
            formatted_path = self._remove_knowledge_formatting_from_reasoning_path(path[0])
            formatted_reasoning_path += formatted_path + self._new_line_char
        # <----
            
        # self._new_line_char.join(
        #         [item[0][0] for item in reasoning_paths])
        
        question = state.get("question", "")
        reasoning_prompt = copy.copy(self.prompt_list.reasoning_prompt)
        reasoning_prompt["prompt"] = reasoning_prompt["prompt"].format(
            question=question,
            reasoning_path=formatted_reasoning_path
        )
        reasoning_res = self.llm_backbone.get_completion(reasoning_prompt)

        logger.debug("<<<<<<<<")
        logger.debug("Reasoning Prompt: {}".format(reasoning_prompt["prompt"]))
        logger.debug("Reasoning Paths: \n{}".format(reasoning_paths))
        logger.debug("Prediction: \n{}".format(reasoning_res))
        logger.debug(">>>>>>>>")

        reasoning_res = reasoning_res.replace("Answer: ", "").strip()

        return reasoning_res

    def planning(
        self,
        state: dict
    ):
        """
        Generate the planning steps for the Beam Search, and the keywords for the Path-RAG
        
        NEW: We adapted the parsing of the planning output to make it more robust if the
        LLM fails to provide a valid JSON object.
        """
        entity = state.get("entity", "")
        if not entity == "":
            entity = entity.text

        question = state.get("question", "")
        plan_prompt = copy.copy(self.prompt_list.plan_prompt)
        plan_prompt["prompt"] = plan_prompt["prompt"].format(
            question=question,
            starting_node=entity
        )

        logger.debug("Plan Prompt: {}".format(plan_prompt))
        plan_res = self.llm_backbone.get_completion(
            plan_prompt).replace("json", "").replace("```", "")
        logger.debug("Plan Response: {}".format(plan_res))
        
        try:
            match = re.search(r'\{.*\}', plan_res, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in plan_res")
            json_str = match.group()
            plan_data = json.loads(json_str)
        except Exception as e:
            logger.error("The LLM was unable to conform to the JSON format, performance will be affected!")
            logger.debug(f"Error in LLM format: {e}. Raw response: {plan_res}")
            
            state["key_words"] = ""
            state["embeddings"] = []
            state["planning_steps"] = ""
            state["declarative_statement"] = ""
            return
        
        key_words = ", ".join(plan_data["keywords"])
        planning_steps = ", ".join(plan_data["planning_steps"])
        declarative_statement = plan_data["declarative_statement"]

        logger.debug("Planning Keywords: {}".format(key_words))
        logger.debug("Planning Steps: {}".format(planning_steps))
        logger.debug("Declarative Statement: {}".format(declarative_statement))

        state["key_words"] = key_words

        embeddings = self.llm_backbone.get_embeddings(key_words)
        state["embeddings"] = embeddings
        state["planning_steps"] = planning_steps
        state["declarative_statement"] = declarative_statement

        # print("planning_steps: ", state["planning_steps"])
        # print("key_words: ", state["key_words"])
        # print("declarative_statement: ", state["declarative_statement"])

    def _try_convert_to_knowledge(self, text: str):
        """
        NEW: Tries to convert the text to a knowledge graph entity
        """
        try:
            knowledge = Knowledge.model_validate_json(text)
            return knowledge
        except Exception:
            return None
        
    def _remove_knowledge_formatting_from_reasoning_path(self, reasoning_path: str):
        """
        NEW: Removes the the Knowledge Model Dumps formatting from the reasoning path
        """
        new_path = []
        for entry in reasoning_path.split(" -> "):
            if "{" in entry and "}" in entry:
                parsed_knowledge = self._try_convert_to_knowledge(entry)
                if parsed_knowledge:
                    new_path.append(parsed_knowledge.text)
                else:
                    new_path.append(entry)
            else:
                new_path.append(entry)
        return " -> ".join(new_path)