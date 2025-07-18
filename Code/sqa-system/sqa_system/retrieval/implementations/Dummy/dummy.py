from typing import Optional
from typing_extensions import override

from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.retrieval import KnowledgeGraphRetriever


class DummyRetriever(KnowledgeGraphRetriever):
    """
    Dummy Retriever that returns a dummy answer.
    """

    @override
    def retrieve_knowledge(
        self,
        query_text: str,
        topic_entity_id: Optional[str],
        topic_entity_value: Optional[str]
    ) -> RetrievalAnswer:
        return RetrievalAnswer(contexts=[], retriever_answer="Dummy Answer")
