import weave
from typing_extensions import override

from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.retrieval import KnowledgeGraphRetrieverFactory
from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.pipe.retrieval.base.retrieval_pipe import RetrievalPipe
from sqa_system.core.data.models.pipe_io_data import PipeIOData
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class KGRetrievalPipe(RetrievalPipe[KGRetrievalConfig]):
    """
    A 'RetrievalPipe' class implementation that is responsible for retrieving data
    from a Knowledge Graph.
    
    Args:
        config (KGRetrievalConfig): The configuration for the Knowledge Graph retrieval.
            It contains the retriever configuration and other parameters.
    """

    def __init__(self, config: KGRetrievalConfig):
        super().__init__(config)
        self._prepare()

    @weave.op()
    @override
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        Uses the prepared Knowledge Graph retriever to retrieve
        contexts based on the initial question provided in the input data.

        The retrieved contexts are then appended to the input data.
        
        Args:
            input_data (PipeIOData): The input data that will be processed.
                It contains the initial question and other relevant information.
                
        Returns:
            PipeIOData: The processed input data with the retrieved contexts
                appended to it.
        """
        retrieval_answer: RetrievalAnswer = self.retriever.retrieve_knowledge(
            query_text=input_data.retrieval_question,
            topic_entity_id=input_data.topic_entity_id,
            topic_entity_value=input_data.topic_entity_value
        )
        if not retrieval_answer:
            logger.warning("No retrieval answer found.")
            return input_data
        input_data.retrieved_context = input_data.retrieved_context or []
        input_data.retrieved_context.extend(retrieval_answer.contexts)
        if retrieval_answer.retriever_answer:
            input_data.generated_answer = retrieval_answer.retriever_answer
        return input_data

    def _prepare(self):
        """
        Prepares the Knowledge Graph retriever for use in the retrieval process.
        This method initializes the retriever based on the provided configuration.
        """
        kg_retriever_manager = KnowledgeGraphRetrieverFactory()
        self.retriever = kg_retriever_manager.create(self.config)
