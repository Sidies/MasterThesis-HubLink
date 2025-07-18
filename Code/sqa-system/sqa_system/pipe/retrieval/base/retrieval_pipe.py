from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from typing_extensions import override
from sqa_system.core.config.models import RetrievalConfig
from sqa_system.pipe.base.pipe import Pipe
from sqa_system.core.data.models.pipe_io_data import PipeIOData
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)

T = TypeVar('T', bound=RetrievalConfig)


class RetrievalPipe(Pipe, ABC, Generic[T]):
    """
    A interface for a pipe that retrieves data from a knowledge base.
    """

    def __init__(self, config: T):
        self.config = config

    @abstractmethod
    @override
    def _process(self, input_data: PipeIOData) -> PipeIOData:
        """
        Classes that inherit from 'RetrievalPipe' must implement this method.

        They should implement the logic for retrieving data from a knowledge base.
        The retrieved data should be appended to the PipeIOData object and returned.
        
        Args:
            input_data (PipeIOData): The input data that will be processed.
                It is extended or manipulated by the pipe.
                
        Returns:
            PipeIOData: The processed input data.
        """
