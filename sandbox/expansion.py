import datetime
import logging
import dspy
import json
import rich
from typing import List, Optional, Literal

from pydantic import BaseModel, Field



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler for logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


model = 'gemini/gemini-2.5-flash-lite-preview-06-17'
# model = 'openai/gpt-4.1-nano'
lm = dspy.LM(model=model, cache=False)
dspy.configure(lm=lm, cache=False)



class BlueLink(BaseModel):
    title: str
    link: str
    snippet: str | None = None
    date: None | str = None


class AnswerBox(BaseModel):
    title: str
    displayed_link: str
    date: None | str = None
    snippet: str
    snippet_highlighted_words: list[str]
    answer: str
    source: str


class DistilledResponse(BaseModel):
    status: Literal['found', 'partial', 'not_found']
    answer: str | None
    reasoning: str | None = None
    resources: list[dict] | None = None
    contradictions: list[dict] | None = None
    suggested_queries: list[str] | None = None
    promising_resources: list[dict] | None = None
    # to_expand: list[BlueLink] | None = None
    # to_discard: list[BlueLink]  | None = None


class SearchEvaluator(dspy.Signature):
    """Evaluates search results, returns answer if found or guidance for further search if not. It carefully evaluates against contradiction/irrelevancy and ambiguity. Any ambiguity/irrelevance/contradiction stated well and complete. Its fine to merge results from different snippets if they are complementing each other"""
    
    # Input fields
    today: str = dspy.InputField(desc="Today's date")
    previous_findings: Optional[str] = dspy.InputField(desc="Previous findings from previous search results")
    query: str = dspy.InputField(desc="The user's search query")
    snippets: List[dict] = dspy.InputField(desc="List of search result snippets with id, title, link, and snippet text")

    # Primary field that determines what other fields will be populated
    status: Literal['found', 'partial', 'not_found'] = dspy.OutputField(desc="'found' if certain/most likely answer found (small inconsistencies accepted), 'partial' if uncertain/contradictory and no clear answer is available, 'not_found' if no answer in snippets")
    
    # Only populated when status is 'found' or 'partial'
    answer: Optional[str] = dspy.OutputField(desc="The extracted answer if status is 'found' or 'partial'. It should be a complete and unambiguous sentence. If not found, return None")
    source_ids: Optional[List[int]] = dspy.OutputField(desc="IDs of snippets supporting the answer")
    
    # Only populated when status is 'partial'
    contradictions: Optional[List[dict]] = dspy.OutputField(desc="Only if status='partial': Contradictions found with conflicting snippet IDs.")
    
    # Only populated when status is 'not_found' or 'partial'
    promising_ids: Optional[List[int]] = dspy.OutputField(desc="Only non empty if status!='found': Up to 3 snippet IDs most relevant for expansion")
    irrelevant_ids: Optional[List[int]] = dspy.OutputField(desc="Only non empty if status!='found': IDs of snippets irrelevant to query")

    suggested_queries: Optional[List[str]] = dspy.OutputField(desc="Only non empty if status!='found': Suggested queries to expand search. Each query should be almost semantically orthogonal to the original query so we maximize the search coverage. They might potentially have very different keywords not necessarily the same query. For example `LMVH` could be expanded To Louis Vuitton Moet Hennesy so we do proper expansion")
    
search_evaluator = dspy.ChainOfThought(SearchEvaluator)

class SearchResult(BaseModel):
    organic: list[BlueLink]
    query: str
    answer_box: None | AnswerBox = Field(default=None, alias='answerBox')


    def _categorize_answer(self, answer: dspy.Prediction) -> DistilledResponse:

        if answer.status == 'found':
            logger.info('Evaluated: found answer for %s query. Answer: %s. Reasoning: %s', self.query, answer.answer, answer.reasoning)
            return DistilledResponse(
                status=answer.status,
                answer=answer.answer,
                reasoning=answer.reasoning,
                resources=[{'id': i, 'title': self.organic[i].title, 'link': self.organic[i].link,
                            'snippet': self.organic[i].snippet} for i in answer.source_ids]
            )
        elif answer.status == 'partial':
            logger.info('Evaluated: found partial answer for %s query. Answer: %s. Reasoning: %s', self.query, answer.answer,
                        answer.reasoning)

            if answer.promising_ids:
                logger.info("Found promising resources to follow up: %s", [{'title': self.organic[i].title, 'link': self.organic[i].link} for i in answer.promising_ids])


            return DistilledResponse(
                status=answer.status,
                answer=answer.answer,
                reasoning=answer.reasoning,
                contradictions=answer.contradictions,
                resources=[{'id': i, 'title': self.organic[i].title, 'link': self.organic[i].link,
                            'snippet': self.organic[i].snippet} for i in answer.source_ids],
                suggested_queries=answer.suggested_queries
            )
        elif answer.status == 'not_found':
            logger.info('Evaluated: No valid answer for %s query. Answer: %s. Reasoning: %s', self.query, answer.answer,
                        answer.reasoning)

            if answer.promising_ids:
                logger.info("Found promising resources to follow up: %s", [{'title': self.organic[i].title, 'link': self.organic[i].link} for i in answer.promising_ids])

            return DistilledResponse(
                status=answer.status,
                answer=answer.answer,
                reasoning=answer.reasoning,
                promising_resources=[{'id': i, 'title': self.organic[i].title, 'link': self.organic[i].link,
                                      'snippet': self.organic[i].snippet} for i in answer.promising_ids],
                suggested_queries=answer.suggested_queries
            )
        else:
            raise ValueError(f"Unknown status: {answer.status}")


    def _form_numbered_links(self):

        return [
            {
                "id": str(i),
                "title": self.organic[i].title,
                "link": self.organic[i].link,
                "snippet": self.organic[i].snippet,
            }
            for i in range(len(self.organic))
        ]


    def evaluate(self) -> DistilledResponse:
        """
        This function tries to extract the answer from resources, if not found it will return the top candidates to expand the links
        """
        logger.info('Evaluating search results... for query: %s', self.query)

        if self.answer_box:
            logger.info('Answer box found for query: %s. Answer: %s, returning', self.query, self.answer_box.answer)
            return DistilledResponse(
                status='found',
                answer=self.answer_box.answer,
                resources=[{'id': -1, 'title': self.answer_box.title, 'link': self.answer_box.displayed_link, 'snippet': 'self.answer_box.snippet'}],
            )

        snippets = self._form_numbered_links()

        logger.info('Evaluating search results with LLM')
        evaluated_answer = search_evaluator(
            today=str(datetime.date.today()),
            previous_findings=None,
            query=self.query,
            snippets=snippets
        )

        return self._categorize_answer(answer=evaluated_answer)


    async def aevaluate(self) -> DistilledResponse:

        if self.answer_box:
            logger.info('Answer box found for query: %s. Answer: %s, returning', self.query, self.answer_box.answer)

            return DistilledResponse(
                status='found',
                answer=self.answer_box.answer,
                resources=[{'id': -1, 'title': self.answer_box.title, 'link': self.answer_box.displayed_link, 'snippet': 'self.answer_box.snippet'}],
            )

            # If not in Answer Box
        snippets = self._form_numbered_links()

        answer = await search_evaluator.acall(
            today=str(datetime.date.today()),
            previous_findings=None,
            query=self.query,
            snippets=snippets
        )

        return self._categorize_answer(answer=answer)


import time
import pathlib

reasoning = dspy.ChainOfThought(SearchEvaluator)


start = time.time()


end = time.time()
print(f"Time taken: {end - start} seconds")

# Search Result is in search_result.json
current_dir = pathlib.Path(__file__).parent


with open(current_dir / "search_result.json", "r") as f:
    search_result_dict = json.load(f)
    search_result_dict['query'] = "What is the length of Pont Alexandre III?"
    search_result = SearchResult(**search_result_dict)

# model = 'gemini/gemini-2.0-flash'
# model = 'openai/gpt-4.1-nano'
# lm = dspy.LM(model=model, cache=False)
# dspy.configure(lm=lm)

# qa = dspy.Predict("question -> answer, confidence_in_percentage")
# result = qa(question="What is the length of Pont Alexandre III?")
#
# rich.print(search_result)

response = search_result.evaluate()
rich.print(response)
#
#
# clear_answer  ={
#     "query": "When was the Eiffel Tower built?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Eiffel Tower - Wikipedia",
#             "link": "https://en.wikipedia.org/wiki/Eiffel_Tower",
#             "snippet": "The Eiffel Tower was built from 1887 to 1889 as the entrance to the 1889 World's Fair."
#         },
#         {
#             "id": "2",
#             "title": "Eiffel Tower History - Official Site",
#             "link": "https://www.toureiffel.paris/en/the-monument/history",
#             "snippet": "Construction work began in January 1887 and was finished on March 31, 1889."
#         }
#     ],
#     "expected_output": {
#         "status": "found",
#         "answer": "The Eiffel Tower was built from 1887 to 1889.",
#         "source_ids": [1, 2],
#         # Other fields should be empty/None
#     }
# }
#
# contradictory = {
#     "query": "How many bones are in the human body?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Human skeleton - Wikipedia",
#             "link": "https://en.wikipedia.org/wiki/Human_skeleton",
#             "snippet": "The adult human skeleton is made up of 206 bones."
#         },
#         {
#             "id": "2",
#             "title": "How many bones in the human body? - Medical News",
#             "link": "https://www.medicalnewstoday.com/articles/human-body-bones",
#             "snippet": "At birth, there are approximately 270 bones in the human body, but many bones fuse together as a child grows up, leaving a total of 206 bones in the adult."
#         },
#         {
#             "id": "3",
#             "title": "Skeletal System Facts",
#             "link": "https://www.factretriever.com/skeletal-system-facts",
#             "snippet": "The human skeletal system consists of 213 bones and over 230 moveable and semi-moveable joints."
#         }
#     ],
#     "expected_output": {
#         "status": "partial",
#         "answer": "The human body has between 206-213 bones in adults, with most sources indicating 206.",
#         "source_ids": [1, 2, 3],
#         "contradictions": [{"description": "Number discrepancy: 206 vs 213 bones", "snippet_ids": [1, 3]}],
#         "promising_ids": [2, 3],
#         "irrelevant_ids": []
#     }
# }
#
# missing = {
#     "query": "What is the population of Wakanda?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Wakanda - Marvel Universe Wiki",
#             "link": "https://marvel.fandom.com/wiki/Wakanda",
#             "snippet": "Wakanda is a fictional African nation in the Marvel Universe, home to the Black Panther and rich in vibranium deposits."
#         },
#         {
#             "id": "2",
#             "title": "Black Panther: Wakanda Forever",
#             "link": "https://en.wikipedia.org/wiki/Black_Panther:_Wakanda_Forever",
#             "snippet": "Black Panther: Wakanda Forever is a 2022 American superhero film based on Marvel Comics featuring the character Shuri."
#         },
#         {
#             "id": "3",
#             "title": "The Geography of Wakanda",
#             "link": "https://www.smithsonianmag.com/arts-culture/geography-wakanda-174123572/",
#             "snippet": "Though fictional, Wakanda is typically located in East Africa, north of Tanzania, and is described as being technologically advanced and isolationist."
#         }
#     ],
#     "expected_output": {
#         "status": "not_found",
#         "promising_ids": [1, 3],
#         "irrelevant_ids": [2],
#         "reasoning": "No population figures are provided for Wakanda in any snippet. Snippets 1 and 3 contain general information about Wakanda that might lead to population data."
#     }
# }
#
# multi_part = {
#     "query": "What are the symptoms of COVID-19?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "COVID-19 symptoms - CDC",
#             "link": "https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html",
#             "snippet": "People with COVID-19 may experience fever or chills, cough, and shortness of breath."
#         },
#         {
#             "id": "2",
#             "title": "COVID-19 symptoms - WHO",
#             "link": "https://www.who.int/health-topics/coronavirus",
#             "snippet": "Most common symptoms of COVID-19 include fever, dry cough, and tiredness. Less common symptoms include aches and pains, sore throat, diarrhea, headache, and loss of taste or smell."
#         },
#         {
#             "id": "3",
#             "title": "Long COVID symptoms",
#             "link": "https://www.nature.com/articles/d41586-022-01453-0",
#             "snippet": "Long COVID can present with fatigue, brain fog, and shortness of breath that persist more than 12 weeks after the initial infection."
#         }
#     ],
#     "expected_output": {
#         "status": "found",
#         "answer": "Common COVID-19 symptoms include fever, cough, shortness of breath, tiredness, aches, sore throat, diarrhea, headache, and loss of taste or smell.",
#         "source_ids": [1, 2],
#         "reasoning": "Combining information from snippets 1 and 2 provides a comprehensive list of symptoms. Snippet 3 is about Long COVID, which is different from initial COVID-19 symptoms."
#     }
# }
#
# irrelevant = {
#     "query": "How to bake sourdough bread?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Bread Making History",
#             "link": "https://www.history.com/news/bread-history",
#             "snippet": "Bread has been a staple food for millennia, with archaeological evidence suggesting it was being baked as far back as 30,000 years ago."
#         },
#         {
#             "id": "2",
#             "title": "Health Benefits of Sourdough",
#             "link": "https://www.healthline.com/nutrition/sourdough-bread",
#             "snippet": "Sourdough bread may be easier to digest and less likely to spike blood sugar levels compared to white bread. It contains more nutrients and fewer phytates."
#         },
#         {
#             "id": "3",
#             "title": "Best Bread Machine Reviews",
#             "link": "https://www.goodhousekeeping.com/bread-machines/",
#             "snippet": "Our top-rated bread machines for 2024 include the Zojirushi Virtuoso Plus and the Breville Custom Loaf, both excellent for beginners."
#         }
#     ],
#     "expected_output": {
#         "status": "not_found",
#         "promising_ids": [2],
#         "irrelevant_ids": [1, 3],
#         "reasoning": "None of the snippets contain actual instructions for baking sourdough bread. Snippet 2 is related to sourdough but doesn't provide baking instructions."
#     }
# }
#
# unit_conv = {
#     "query": "How tall is Mount Everest?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Mount Everest - Wikipedia",
#             "link": "https://en.wikipedia.org/wiki/Mount_Everest",
#             "snippet": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Its elevation of 8,848.86 m (29,031.7 ft) was most recently established in 2020 by the Chinese and Nepali authorities."
#         },
#         {
#             "id": "2",
#             "title": "Facts about Mount Everest",
#             "link": "https://www.livescience.com/23359-mount-everest.html",
#             "snippet": "At 29,029 feet (8,848 meters), Mount Everest is the highest mountain on Earth."
#         },
#         {
#             "id": "3",
#             "title": "Why Mount Everest keeps changing its height",
#             "link": "https://www.nationalgeographic.com/science/article/everest-height",
#             "snippet": "In December 2020, China and Nepal jointly announced that the world's highest peak is 8,848.86 meters (29,031.7 feet) above sea level, which is slightly more than Nepal's previous measurement."
#         }
#     ],
#     "expected_output": {
#         "status": "found",
#         "answer": "Mount Everest is 8,848.86 meters (29,031.7 feet) tall, as established in 2020 by Chinese and Nepali authorities.",
#         "source_ids": [1, 3],
#         "reasoning": "Snippets 1 and 3 contain the most recent measurement from 2020. Snippet 2 has a slightly different value that appears to be rounded or outdated."
#     }
# }
#
# temporal = {
#     "query": "Who is the current President of France?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Emmanuel Macron - Wikipedia",
#             "link": "https://en.wikipedia.org/wiki/Emmanuel_Macron",
#             "snippet": "Emmanuel Jean-Michel Frédéric Macron (born 21 December 1977) is a French politician who has been serving as the president of France since 14 May 2017."
#         },
#         {
#             "id": "2",
#             "title": "French Presidential Election 2022",
#             "link": "https://www.france24.com/en/europe/20220424",
#             "snippet": "Emmanuel Macron won a second term as president of France, with 58.2% of the vote in the second round, according to exit polls, defeating Marine Le Pen."
#         },
#         {
#             "id": "3",
#             "title": "List of Presidents of France",
#             "link": "https://www.elysee.fr/en/presidents-of-the-republic",
#             "snippet": "The current President of the French Republic is Emmanuel Macron, who succeeded François Hollande on 14 May 2017."
#         }
#     ],
#     "expected_output": {
#         "status": "found",
#         "answer": "Emmanuel Macron is the current President of France, serving since May 14, 2017.",
#         "source_ids": [1, 2, 3],
#         "reasoning": "All snippets consistently identify Emmanuel Macron as the current President of France."
#     }
# }
#
# ambiguous = {
#     "query": "When did the Queen die?",
#     "snippets": [
#         {
#             "id": "1",
#             "title": "Queen Elizabeth II - Wikipedia",
#             "link": "https://en.wikipedia.org/wiki/Elizabeth_II",
#             "snippet": "Queen Elizabeth II (Elizabeth Alexandra Mary; 21 April 1926 – 8 September 2022) was Queen of the United Kingdom and other Commonwealth realms from 6 February 1952 until her death in 2022."
#         },
#         {
#             "id": "2",
#             "title": "Queen Victoria Death",
#             "link": "https://www.history.com/this-day-in-history/queen-victoria-dies",
#             "snippet": "Queen Victoria, the longest-reigning monarch in British history, died at the age of 81 on January 22, 1901."
#         },
#         {
#             "id": "3",
#             "title": "Freddie Mercury - Death",
#             "link": "https://www.biography.com/musicians/freddie-mercury",
#             "snippet": "Freddie Mercury, lead vocalist of the rock band Queen, died of bronchial pneumonia resulting from AIDS on November 24, 1991."
#         }
#     ],
#     "expected_output": {
#         "status": "partial",
#         "answer": "The query is ambiguous: Queen Elizabeth II died on September 8, 2022; Queen Victoria died on January 22, 1901; Freddie Mercury of the band Queen died on November 24, 1991.",
#         "source_ids": [1, 2, 3],
#         "contradictions": [{"description": "Multiple 'Queens' referenced: Elizabeth II, Victoria, and the band Queen", "snippet_ids": [1, 2, 3]}],
#         "reasoning": "The query 'When did the Queen die?' is ambiguous as it could refer to different queens or people associated with 'Queen'."
#     }
# }

# response = reasoning(**ambiguous)
# rich.print(response)
