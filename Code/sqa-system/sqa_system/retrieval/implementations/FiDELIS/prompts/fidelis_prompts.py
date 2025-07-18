reasoning_prompt = {
   "system": "Given a question and the associated retrieved reasoning path from a knowledge graph, you are asked to answer the following question based on the reasoning path. Only return the answer to the question directly. Do not add further comments.",
   "examples": [
      {"role": "user", "content": "Question: Does the paper 'Infrastructure-as-Code for Data-Intensive Architectures: A Model-Driven Development Approach' use tool support? \nReasoning path: Infrastructure-as-Code for Data-Intensive Architectures: A Model-Driven Development Approach->Tool Support"}, 
      {"role": "assistant", "content": "Answer: Yes, the paper indicates that it uses tool support."},
      {"role": "user", "content": "Question: Which publications have the evaluation method Questionnaire?\n Reasoning path: Evaluation Method Questionnaire"},
      {"role": "assistant", "content": "Answer: The evaluation method Questionnaire is discussed in the following publications: 1. Technical Architectures for Automotive Systems, 2. Architectural Assumptions and Their Management in Industry – An Exploratory Study, 3. An Exploratory Study of Naturalistic Decision Making in Complex Software Architecture Environments"},
      {"role": "user", "content": "Question: How often is the sub-property Satisfaction used in comparison to the sub-property Context coverage with the research object named Technical Architectures?\nReasoning path: Research Object: Technical Architectures->Sub-Property: Satisfaction->Sub-Property: Context coverage"},
      {"role": "assistant", "content": "Answer: The sub-property Satisfaction is used two times, while the sub-property Context coverage is used once."},
      {"role": "user", "content": "Question: What research objects, that have been published by the author Uwe Zdun, have no evaluation method?\nReasoning path: Author: Uwe Zdun->Evaluation Method: False->Research Objects"},
      {"role": "assistant", "content": "Answer: The research objects published by Uwe Zdun that have no evaluation method are: 1. Technical Architectures and 2. Architectural Assumptions."},
      {"role": "user", "content": "Question: How many evaluation methods does the paper titled 'Continuous Architecture: Towards the Goldilocks Zone and Away from Vicious Circles' have?\nReasoning Path: Paper: Continuous Architecture: Towards the Goldilocks Zone and Away from Vicious Circles->Evaluation Methods"},
      {"role": "assistant", "content": "Answer: The paper titled 'Continuous Architecture: Towards the Goldilocks Zone and Away from Vicious Circles' has 3 evaluation methods."},
   ],
   "prompt": "Given a question and the associated retrieved reasoning path from a knowledge graph, you are asked to answer the following question based on the reasoning path and your knowledge. \nQuestion: {question} \nReasoning path: {reasoning_path} \n"
}

beam_search_prompt = {
   "system": "Given a question and the starting entity from a knowledge graph, you are asked to retrieve reasoning paths from the given reasoning paths that are useful for answering the question. Each reasoning path has a index attached to it. You are only allowed to return the index as a list inside of brackets e.g. [1, 2, 3].",
   "examples": [
      {"role": "user", "content": "Question: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?\n To proceed, the starting entity is Software Architecture and Design\n Out of the following candidates for the next step, choose the best 5 with the highest probability to lead to a useful reasoning path for answering the question. \nCandidates: 1: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> Second Research Object \n 2: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> venue -> European Conference on Software Architecture (ECSA)\n 3: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> contribution -> First Research Object 2\n 4: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> contribution -> Research Level 2\n 5: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> contribution -> Paper Class 2\n 6: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> contribution -> Evidence 2 \n7: Software Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> contribution -> Validity 2"},
      {"role": "assistant", "content": "Answer: [1, 3, 5, 6, 7]"},
   ],
   "prompt": "Considering the planning context {plan_context} and the given question {question}, \nyou are asked to choose the best {beam_width} reasoning paths from the following candidates with the highest probability to lead to a useful reasoning path for answering the question. {reasoning_paths}. \n\nOnly return the index of the {beam_width} selected reasoning paths in a list."
}

# planning + keywords detection
plan_prompt = {
   "system": "You are a helpful assistant designed to output JSON that aids in navigating a knowledge graph to answer a provided question. The response should include the following keys: (1) 'keywords': an exhaustive list of keywords or relation names that you would use to find the reasoning path from the knowledge graph to answer the question. Aim for maximum coverage to ensure no potential reasoning paths will be overlooked; (2) 'planning_steps': a list of detailed steps required to trace the reasoning path with. Each step should be a string instead of a dict. (3) 'declarative_statement': a string of declarative statement that can be transformed from the given query, For example, convert the question 'What papers investigate the evaluation method Interview?' into the statement 'The papers that investigate the evaluation method Interview are: *placeholder*' leave the *placeholder* unchanged; Ensure the JSON object clearly separates these components.",
   "examples": [],
   "prompt": "{question}?"
}

reasoning_path_parser_prompt = {
   "system": "You are asked to transfer the reasoning path to a statment, for example, convert the reasoning path Software Architecture and Design -> research field -> Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models -> research object -> Architecture Description Language -> evaluation -> Evaluation Method List -> evaluation method -> Interview to the statement 'The paper Availability-Driven Architectural Change Propagation Through Bidirectional Model Transformations Between UML and Petri Net Models is evaluated by the evaluation method Interview'. Important: Only provide the statement without any additional information. Just output the statement!",
   "examples": [],
   "prompt": "{reasoning_path}",
}

terminals_prune_single_prompt = {
   "system": "Given a question and the starting entity from a knowledge graph, you are asked to answer whether it's sufficient for you to answer the question with the following reasoning path. For each reasoning path, respond with 'Yes' if it is sufficient, and 'No' if it is not. Your response should be either 'Yes' or 'No', corresponding to reasoning path.",
   "examples": [
      {"role": "user", "content": "Question: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?\n Reasoning path: \nSoftware Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> Second Research Object -> Reference Architecture -> Evaluation Method List -> Evaluation Method -> Interview"},
      {"role": "assistant", "content": "Answer: Yes"},
      {"role": "user", "content": "Question: What is the venue of the paper ACE: Easy Deployment of Field Optimization Experiments\nReasoning path: \nSoftware Architecture and Design -> research field -> ACE: Easy Deployment of Field Optimization Experiments -> venue -> European Conference on Software Architecture (ECSA)"},
      {"role": "assistant", "content": "Answer: Yes"},
      {"role": "user", "content": "How often is the sub-property Satisfaction used in comparison to the sub-property Context coverage with the research object named Technical Architectures?\nReasoning path: ACE: Easy Deployment of Field Optimization Experiments -> contribution -> Evidence 2"},
      {"role": "assistant", "content": "Answer: No"},
   ],
   "prompt": "Considering the planning context {plan_context} and the given question {question}, you are asked to answer whether it's sufficient for you to answer the question with the following reasoning path. {reasoning_path}. Verify if it is sufficient to answer the question with each of the provided reasoning path. Your response should either 'Yes' or 'No', corresponding to reasoning path. \nAnswer:"
}

deductive_verifier_prompt = {
   "system": "You are asked to verify whether the reasoning step follows deductively from the question and the current reasoning path in a deductive manner. If yes return 'yes', if no, return 'no'",
   "examples": [],
   "prompt": "Whether the conclusion \'{declarative_statement}\' can be deduced from \'{parsed_reasoning_path}\', if yes, return 'yes', if no, return 'no'. Only return 'yes' if the conclusion is plausible."
}