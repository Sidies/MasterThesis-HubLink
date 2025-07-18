"""
The following file is a implementation of the Think-on-Graph retriever.
The repository can be found here: https://github.com/IDEA-FinAI/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67

Their code has been adapted to work with this project.
"""

# ----> NEW
# We adapted the prompt examples to our use case.
extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?
Topic Entity: Software Architecture and Design
Relations: contribution; First Research Object; Paper Class; Research Level; Evidence; Validity;
A: 1. {First Research Object (Score: 0.4))}: This relation is highly relevant as research objects are typicall evaluated by an evaluation method.
2. {contribution (Score: 0.3)}: This relation is also very relevant as inside of the contributions there are often evaluation methods mentioned.
3. {Validity (Score: 0.2)}: This relation is less relevant but still could provide information about the evaluation methods used in the paper.

Q: """
# <----
# ----> NEW
# We adapted the prompt examples to our use case.
score_entity_candidates_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?
Relation: First Research Object
Entites: Evaluation; Object; Validity; Evidence; Paper Class; Research Level
Score: 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
The most relevant entity that could lead us to the answer of the question is the Evaluaton entity. The other entities are less relevant and do not contribute to the question. Therefore, we assign a score of 1.0 to the Evaluation entity and 0.0 to the others.

Q: {}
Relation: {}
Entites: """
# <----
# ----> NEW
# We adapted the prompt examples to our use case.
answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?
Knowledge Triplets: Evaluation Method List, Evaluation Method, Interview
Evaluation Method List, Evaluation Method, Survey
A: The evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments' are: Interview and Survey

Q: What is the paper class of the paper 'A Blockchain-Based Micro Economy Platform for Distributed Infrastructure Initiatives'?
Knowledge Triplets: paper class, proposal of solution, true
paper class, philosophical paper, false
paper class, validation research, true
A: Based on the given knowledge triplets, the paper class of the paper 'A Blockchain-Based Micro Economy Platform for Distributed Infrastructure Initiatives' is proposal of solution and validation research. 

Q: What are the research objects of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment'?
Knowledge Triplets: Research Object Entity, Name, Architecture Description Language
Research Object Entity, Name, Reference Architecture
A: Based on the given knowledge triplets, the research objects of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment' are: Architecture Description Language and Reference Architecture.

Q: Who are the authors of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment'?
Knowledge Triplets: authors list, has list element, Patricia Lago
authors list, has list element, Roberto Verdecchia
authors list, has list element, Ivano Malavolta
A: Based on the given knowledge triplets, the authors of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment' are: Patricia Lago, Roberto Verdecchia and Ivano Malavolta.

Q: {}
"""
# <----
# ----> NEW
# We adapted the prompt examples to our use case.
prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Who are the authors of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment'?
Knowledge Triplets: Research Object Entity, Name, Architecture Description Language
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The knowledge triplet only provides information about the research object entity and its name, but it does not provide any information about the authors of the paper. Therefore, additional knowledge or triplets are needed to answer the question.

Q: What are the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments'?
Knowledge Triplets: Evaluation Method List, Evaluation Method, Interview
Evaluation Method List, Evaluation Method, Survey
A: {Yes}. Based on the given knowledge triplets, the evaluation methods of the paper 'ACE: Easy Deployment of Field Optimization Experiments' are: Interview and Survey. Therefore, the answer to the question is {Interview, Survey}.

Q: What is the paper class of the paper 'A Blockchain-Based Micro Economy Platform for Distributed Infrastructure Initiatives'?
Knowledge Triplets: Evidence 1, Uses Tool Support, No tool used
Evidence 1, Uses Input Data, No input data used
A: {No}. Based on the given knowledge triplets, the paper class of the paper 'A Blockchain-Based Micro Economy Platform for Distributed Infrastructure Initiatives' is not explicitly mentioned. The knowledge triplets only provide information about the evidence related to tool support and input data used in the paper. Therefore, additional knowledge or triplets are needed to answer the question.

Q: What are the research objects of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment'?
Knowledge Triplets: Research Object Entity, Name, Architecture Description Language
Research Object Entity, Name, Reference Architecture
A: {Yes}. Based on the given knowledge triplets, the research objects of the paper 'On the Understandability of Semantic Constraints for Behavioral Software Architecture Compliance: A Controlled Experiment' are: Architecture Description Language and Reference Architecture. Therefore, the answer to the question is {Architecture Description Language, Reference Architecture}.

Q: What are the threats to validity of the paper 'Towards Micro Service Architecture Recovery: An Empirical Study'?
Knowledge Triplets: Paper Class 1, paper class, Validation Research
A: {No}. Based on the given knowledge triplets, the threats to validity of the paper 'Towards Micro Service Architecture Recovery: An Empirical Study' are not explicitly mentioned. The knowledge triplet only provides information about the paper class, but it does not provide any information about the threats to validity. Therefore, additional knowledge or triplets are needed to answer the question.

"""
# <----
generate_directly = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""

score_entity_candidates_prompt_wiki = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: Staten Island Summer, starred what actress who was a cast member of "Saturday Night Live"?
Relation: cast member
Entites: Ashley Greene; Bobby Moynihan; Camille Saviola; Cecily Strong; Colin Jost; Fred Armisen; Gina Gershon; Graham Phillips; Hassan Johnson; Jackson Nicoll; Jim Gaffigan; John DeLuca; Kate Walsh; Mary Birdsong
Score: 0.0, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0
To score the entities\' contribution to the question, we need to determine which entities are relevant to the question and have a higher likelihood of being the correct answer.
In this case, we are looking for an actress who was a cast member of "Saturday Night Live" and starred in the movie "Staten Island Summer." Based on this information, we can eliminate entities that are not actresses or were not cast members of "Saturday Night Live."
The relevant entities that meet these criteria are:\n- Ashley Greene\n- Cecily Strong\n- Fred Armisen\n- Gina Gershon\n- Kate Walsh\n\nTo distribute the scores, we can assign a higher score to entities that are more likely to be the correct answer. In this case, the most likely answer would be an actress who was a cast member of "Saturday Night Live" around the time the movie was released.
Based on this reasoning, the scores could be assigned as follows:\n- Ashley Greene: 0\n- Cecily Strong: 0.4\n- Fred Armisen: 0.2\n- Gina Gershon: 0\n- Kate Walsh: 0.4

Q: {}
Relation: {}
Entites: """

prompt_evaluate_wiki="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: {Yes}. Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: {No}. Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned. Therefore, additional knowledge about the specific compounds and their conversion to psilocin is required to answer the question.

Q: Which tennis player is younger, John Newcombe or Květa Peschke?
Knowledge Triplets: Květa Peschke, date of birth, +1975-07-09T00:00:00Z, 
John Newcombe, date of birth, +1944-05-23T00:00:00Z,
John Newcombe, country of citizenship, Australia
A: {Yes}. Based on the given knowledge triplets and my knowledge, John Newcombe was born on May 23, 1944, and Květa Peschke was born on July 9, 1975. Therefore, {Květa Peschke} is younger than John Newcombe.

Q: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?
Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center
San Antonio Spurs, home venue, Alamodome
San Antonio Spurs, home venue, Fort Worth Convention Center
AT&T Center, occupant, San Antonio Spurs
Fort Worth Convention Center, located in the administrative territorial entity, Texas
Fort Worth Convention Center, occupant, San Antonio Spurs
A: {Yes}. Based on the given knowledge triplets and my knowledge, Mychal George Thompson played home games with the San Antonio Spurs at the AT&T Center. Therefore, the answer to the question is {AT&T Center}.

"""
extract_relation_prompt_wiki = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
Q: Mesih Pasha's uncle became emperor in what year?
Topic Entity: Mesih Pasha
Relations:
1. wiki.relation.child
2. wiki.relation.country_of_citizenship
3. wiki.relation.date_of_birth
4. wiki.relation.family
5. wiki.relation.father
6. wiki.relation.languages_spoken, written_or_signed
7. wiki.relation.military_rank
8. wiki.relation.occupation
9. wiki.relation.place_of_death
10. wiki.relation.position_held
11. wiki.relation.religion_or_worldview
12. wiki.relation.sex_or_gender
13. wiki.relation.sibling
14. wiki.relation.significant_event
A: 1. {wiki.relation.family (Score: 0.5)}: This relation is highly relevant as it can provide information about the family background of Mesih Pasha, including his uncle who became emperor.
2. {wiki.relation.father (Score: 0.4)}: Uncle is father's brother, so father might provide some information as well.
3. {wiki.relation.position held (Score: 0.1)}: This relation is moderately relevant as it can provide information about any significant positions held by Mesih Pasha or his uncle that could be related to becoming an emperor.

Q: Van Andel Institute was founded in part by what American businessman, who was best known as co-founder of the Amway Corporation?
Topic Entity: Van Andel Institute
Relations:
1. wiki.relation.affiliation
2. wiki.relation.country
3. wiki.relation.donations
4. wiki.relation.educated_at
5. wiki.relation.employer
6. wiki.relation.headquarters_location
7. wiki.relation.legal_form
8. wiki.relation.located_in_the_administrative_territorial_entity
9. wiki.relation.total_revenue
A: 1. {wiki.relation.affiliation (Score: 0.4)}: This relation is relevant because it can provide information about the individuals or organizations associated with the Van Andel Institute, including the American businessman who co-founded the Amway Corporation.
2. {wiki.relation.donations (Score: 0.3)}: This relation is relevant because it can provide information about the financial contributions made to the Van Andel Institute, which may include donations from the American businessman in question.
3. {wiki.relation.educated_at (Score: 0.3)}: This relation is relevant because it can provide information about the educational background of the American businessman, which may have influenced his involvement in founding the Van Andel Institute.

Q: """

answer_prompt_wiki = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your own knowledge.
Q: Viscount Yamaji Motoharu was a general in the early Imperial Japanese Army which belonged to which Empire?
Knowledge Triplets: Imperial Japanese Army, allegiance, Emperor of Japan
Yamaji Motoharu, allegiance, Emperor of Japan
Yamaji Motoharu, military rank, general
A: Based on the given knowledge triplets and my knowledge, Viscount Yamaji Motoharu, who was a general in the early Imperial Japanese Army, belonged to the Empire of Japan. Therefore, the answer to the question is {Empire of Japan}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: psilocybin, described by source, Opium Law,
psilocybin, found in taxon, Gymnopilus purpuratus,
psilocybin, found in taxon, Gymnopilus spectabilis, 
Opium Law, part of, norcodeine (stereochemistry defined), 
Gymnopilus purpuratus, edibility, psychoactive mushroom,
Gymnopilus spectabilis, parent taxon, Gymnopilus
A: Based on the given knowledge triplets and my knowledge, the specific psychedelic compound found in the Psilocybin genus mushroom that is converted to psilocin by the body is not explicitly mentioned. Therefore, additional knowledge about the specific compounds and their conversion to psilocin is required to answer the question.

Q: Which tennis player is younger, John Newcombe or Květa Peschke?
Knowledge Triplets: Květa Peschke, date of birth, +1975-07-09T00:00:00Z, 
John Newcombe, date of birth, +1944-05-23T00:00:00Z,
John Newcombe, country of citizenship, Australia
A: Based on the given knowledge triplets and my knowledge, John Newcombe was born on May 23, 1944, and Květa Peschke was born on July 9, 1975. Therefore, {Květa Peschke} is younger than John Newcombe.

Q: At what stadium did Mychal George Thompson play home games with the San Antonio Spurs?
Knowledge Triplets: San Antonio Spurs, home venue, AT&T Center
San Antonio Spurs, home venue, Alamodome
San Antonio Spurs, home venue, Fort Worth Convention Center
AT&T Center, occupant, San Antonio Spurs
Fort Worth Convention Center, located in the administrative territorial entity, Texas
Fort Worth Convention Center, occupant, San Antonio Spurs
A: Based on the given knowledge triplets and my knowledge, Mychal George Thompson played home games with the San Antonio Spurs at the AT&T Center. Therefore, the answer to the question is {AT&T Center}.

Q: {}
"""

cot_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}."""