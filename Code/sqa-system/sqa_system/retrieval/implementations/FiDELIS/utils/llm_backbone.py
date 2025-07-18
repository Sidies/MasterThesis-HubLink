
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.config.models import LLMConfig, EmbeddingConfig

logger = get_logger(__name__)

class LLM_Backbone():
    
    def __init__(self, llm_config: LLMConfig, embedding_config: EmbeddingConfig):
        provider = LLMProvider()
        self.llm = provider.get_llm_adapter(llm_config)
        self.embedding_model = provider.get_embeddings(embedding_config)
        self.max_attempt = 5 # number of attempts to get the completion
        
    def get_embeddings(self, texts:list):
        """
        NEW: Adapted to work with the interfaces of the SQA System.
        """
        embeddings = []
        texts_per_batch = 2000
        text_chunks = [texts[i:i + texts_per_batch] for i in range(0, len(texts), texts_per_batch)]
        
        attempt = 0
        while attempt < self.max_attempt:
            try:
                for chunk in text_chunks:
                    chunk_embeddings = self.embedding_model.embed(chunk)
                    embeddings.extend(chunk_embeddings)
                return embeddings
            except Exception as e:
                logger.error(f"Error occurred: {e}")
                attempt += 1
                time.sleep(1)
                
                
    def get_completion(self, prompt: dict) -> str:
        """
        NEW: Adapted to work with the interfaces of the SQA System.
        """
        messages = [
            SystemMessage(content=prompt["system"]),
            *[HumanMessage(content=example["content"]) for example in prompt["examples"]],
            HumanMessage(content=prompt["prompt"])
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
            
        attempt = 0
        while attempt < self.max_attempt:
            try:
                answer = self.llm.generate(str(prompt))
                return answer.content
            except Exception as e:
                logger.error(f"Error occurred: {e}")
                attempt += 1
                time.sleep(1)