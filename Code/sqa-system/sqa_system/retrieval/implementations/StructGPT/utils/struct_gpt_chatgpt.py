import json
import re
import time

import openai

from langchain_core.prompts import ChatPromptTemplate

from sqa_system.core.language_model.llm_provider import LLMProvider

from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)

class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.idx_mapping = {"0": "first", "1": "second", "2": "third", "3": "fourth", "4": "fifth", "5": "sixth",
                            "6": "seventh", "7": "eighth", "8": "ninth", "9": "tenth"}
        # ----> NEW
        self.llm_adapter = LLMProvider().get_llm_adapter(self.args.llm_config)
        # <----

    def get_response(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            # ----> NEW
            # Added logging
            logger.debug("query API to get message:\n%s" % message['content'])
            # <----
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result(message)
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type, tpe_name)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result(message, turn_type)
        return response

    def get_response_v1(self, input_text, turn_type, tpe_name=None):
        if self.args.debug:
            message = self.create_message_v1(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            # ----> NEW
            # Added logging
            logger.debug("query API to get message:\n%s" % message['content'])
            # <----
            # message = self.query_API_to_get_message(self.history)
            # self.history.append(message)
            # response = self.parse_result(message)
            response = input("input the returned response:")
        else:
            message = self.create_message_v1(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            response = self.parse_result_v1(message, turn_type)
        return response

    def create_message(self, input_text, turn_type, tpe_name):
        if turn_type == "initial":  # the initial query
            instruction = self.prompt[turn_type]['instruction']
            template = self.prompt[turn_type]['init_template']
            self.question = input_text
            input_text = instruction + template.format(question=input_text, tpe=tpe_name)
        elif turn_type == "continue_template":
            input_text = self.prompt[turn_type]
        elif turn_type == "question_template":
            template = self.prompt[turn_type]
            input_text = template.format(idx=self.idx_mapping[input_text])
        elif turn_type == "answer_template":
            template = self.prompt[turn_type]
            if len(input_text) > 0:
                input_text = template["valid"].format(facts=input_text)
            else:
                input_text = template["invalid"]
        elif turn_type == "final_query_template":
            template = self.prompt[turn_type]
            input_text = template.format(question=self.question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def create_message_v1(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            question, tpe, can_rels = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels)
        elif turn_type == "ask_question":
            template = self.prompt['ask_question']
            idx, relations = input_text
            idx = self.idx_mapping[idx]
            input_text = template.format(idx=idx, relations=relations)
        elif turn_type == "ask_answer":
            facts = input_text
            template = self.prompt['ask_answer']
            input_text = template.format(facts=facts)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "condition":
            input_text = self.prompt['continue_template']['condition']
        elif turn_type == "continue":
            input_text = self.prompt['continue_template']['continue']
        elif turn_type == "stop":
            input_text = self.prompt['continue_template']['stop']
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            question, can_rels = input_text
            input_text = template.format(question=question, relations=can_rels)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        """
        NEW: Changed function to query using the LLMAdapter interface
        of the SQA System
        """
        number_retries = 4
        while number_retries > 0:
            try:
                prompt = ChatPromptTemplate.from_messages(messages)
                logger.debug(f"LLM Prompt: {str(prompt)}")
                response = self.llm_adapter.generate(str(prompt))
                return {"role": "assistant", "content": response.content}
            except openai.RateLimitError:
                logger.debug('openai.RateLimitError\nRetrying...')
                time.sleep(30)
            except openai.APIError as e:
                logger.debug(f'OpenAI API error: {e}\nRetrying...')
                time.sleep(20)
            except Exception as e:
                logger.debug(f'Unexpected error: {e}\nRetrying...')
                time.sleep(20)
            number_retries -= 1

    def parse_result(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["initial", "question_template"]:
            if "should be" in content:
                content = content.split("should be")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        # ----> NEW
                        # Added logging
                        logger.debug("Not exactly parse, we directly use content: %s" % content)
                        # <----

        return content

    def parse_result_v1(self, result, turn_type):
        content = result['content'].strip()
        if turn_type in ["ask_question", "continue"]:
            if "the simple question:" in content:
                content = content.split("the simple question:")[1].strip()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                else:
                    matchObj = re.search(r'"(.*?)"', content)
                    if matchObj is not None:
                        content = matchObj.group()
                        content = content[1:-1]
                    else:
                        content = content.strip().strip('"')
                        # ----> NEW
                        # Added logging
                        logger.debug("Not exactly parse, we directly use content: %s" % content)
                        # <----

        return content

    def parse_result_v2(self, result, turn_type):
        content = result['content'].strip()

        return content

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []

    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]

    def get_response_v2(self, input_text, turn_type):
        message = self.create_message_v2(input_text, turn_type)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        message = self.query_API_to_get_message(self.history_messages)
        self.history_messages.append(message)
        self.history_contents.append(message['content'])
        response = message['content'].strip()

        return response

    def create_message_v2(self, input_text, turn_type):
        if turn_type == "instruction":  # the initial query
            instruction = self.prompt['instruction']
            input_text = instruction
        # ykm
        # elif turn_type == "init_relation_rerank":
        #     template = self.prompt['init_relation_rerank']
        #     can_rels, question, tpe, hop = input_text
        #     if hop == 1:
        #         hop = "first"
        #     elif hop == 2:
        #         hop = "second"
        #     elif hop == 3:
        #         hop = "third"
        #     input_text = template.format(question=question, tpe=tpe, relations=can_rels, hop=hop)
        elif turn_type == "init_relation_rerank":
            template = self.prompt['init_relation_rerank']
            can_rels, question, tpe = input_text
            input_text = template.format(question=question, tpe=tpe, relations=can_rels, width=self.args.width)
        elif turn_type == "constraints_flag":
            template = self.prompt['constraints_flag']
            question, tpe, selected_relations = input_text
            if len(selected_relations) > 1:
                selected_relations = "are " + ", ".join(selected_relations)
            else:
                selected_relations = "is " + ", ".join(selected_relations)
            input_text = template.format(question=question, tpe=tpe, selected_relations=selected_relations)
        elif turn_type == "ask_final_answer_or_next_question":
            question, serialized_facts = input_text
            template = self.prompt['ask_final_answer_or_next_question']
            input_text = template.format(facts=serialized_facts, question=question)
        elif turn_type == "choose_constraints":
            question, relation_tails, tpe_name = input_text
            template = self.prompt['choose_constraints']
            input_text = template.format(question=question, relation_tails=relation_tails, tpe=tpe_name)
        elif turn_type == "final_query_template":
            template = self.prompt['final_query_template']
            input_text = template.format(question=input_text)
        elif turn_type == 'relation_rerank':
            template = self.prompt['relation_rerank']
            can_rels, question, tpe, selected_relations = input_text
            # 暂时注释掉
            selected_relations = "".join(selected_relations)
            # ----> NEW
            # We removed the "selected relations" from the prompt as we found it to be confusing for the LLM which may be because of our data.
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                        #  selected_relations=selected_relations,
                                         width=self.args.width)
            # <----
        elif turn_type == 'relation_rerank_2hop':
            template = self.prompt['relation_rerank_2hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            sub_question = ", ".join(sub_question)
            selected_relations = ", ".join(selected_relations)
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=sub_question, first_relation=selected_relations)
        elif turn_type == 'relation_rerank_3hop':
            template = self.prompt['relation_rerank_3hop']
            can_rels, question, tpe, sub_question, selected_relations = input_text
            first_sub_question = sub_question[0]
            second_sub_question = sub_question[1]
            fisrt_relation = selected_relations[0]
            second_relation = selected_relations[1]
            input_text = template.format(question=question, relations=can_rels, tpe=tpe,
                                         first_sub_question=first_sub_question, first_relation = fisrt_relation,
                                         second_sub_question=second_sub_question, second_relation=second_relation)
        elif turn_type == 'direct_ask_final_answer':
            template = self.prompt['direct_ask_final_answer']
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == 'final_answer_organize':
            template = self.prompt['final_answer_organize']
            input_text = template
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message