import os
from typing import ClassVar, List
import re
import csv
import io

from lightrag import (
    LightRAG,
    QueryParam,
)
from lightrag.prompt import PROMPTS
from lightrag.llm.openai import (
    openai_complete_if_cache,
    openai_embed,
    GPTKeywordExtractionFormat,
    AsyncOpenAI,
    wrap_embedding_func_with_attrs
)
from lightrag.llm.ollama import (
    ollama_model_complete,
    ollama_embedding
)
from lightrag.utils import EmbeddingFunc

import numpy as np
from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.config.models import DocumentRetrievalConfig
from sqa_system.core.data.secret_manager import SecretManager
from sqa_system.core.data.dataset_manager import DatasetManager
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.models import Context, ContextType
from sqa_system.core.config.models import DatasetConfig
from sqa_system.core.language_model.enums.llm_enums import EndpointType
from sqa_system.core.logging.logging import get_logger

from ...base.document_retriever import DocumentRetriever

logger = get_logger(__name__)

DEFAULT_LLM_MODEL_TYPE = "googleapi"
DEFAULT_LLM_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_EMBEDDING_MODEL_TYPE = "googleapi"
DEFAULT_EMBEDDING_MODEL_NAME = "models/text-embedding-004"
DEFAULT_GENERATION_CONFIG = LLMConfig(
    endpoint=EndpointType.OPENAI.value,
    max_tokens=1024,
    additional_params={},
    temperature=0.1,
    name_model="gpt-4o-mini"
)


class LightRag(DocumentRetriever):
    """
    This is the implementation of the LightRAG retriever which has been proposed by
    Guo et al. in their paper "LightRAG: Simple and Fast Retrieval-Augmented Generation"
    
    The paper is accessible here: https://arxiv.org/abs/2410.05779
    And the repository is accessible here: https://github.com/HKUDS/LightRAG
    
    They already provide a general implementation of the retriever which can be accessed on
    PyPi: https://pypi.org/project/lightrag-hku/
    
    We adapted this implementation to work with our framework, however this also means
    that some things we were not able to seamingly integrate into our framework.
    As such, our LLM Adapter 
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        # Light RAG has its own implementation for the LLM model which is why we can't
        # Use our LLM adapter here.
        AdditionalConfigParameter(
            name="llm_model_type",
            description="The type of the LLM model to use",
            param_type=str,
            available_values=["openai", "ollama", "googleapi"],
            default_value=DEFAULT_LLM_MODEL_TYPE,
        ),
        AdditionalConfigParameter(
            name="llm_model_name",
            description="The name of the LLM model to use",
            param_type=str,
            available_values=[],
            default_value=DEFAULT_LLM_MODEL_NAME,
        ),
        AdditionalConfigParameter(
            name="llm_embedding_model_type",
            description="The type of the LLM model to use for embeddings",
            param_type=str,
            available_values=["openai", "googleapi", "ollama"],
            default_value=DEFAULT_EMBEDDING_MODEL_TYPE,
        ),
        AdditionalConfigParameter(
            name="llm_embedding_model_name",
            description="The name of the LLM model to use for embeddings",
            param_type=str,
            available_values=[],
            default_value=DEFAULT_EMBEDDING_MODEL_NAME,
        ),
        AdditionalConfigParameter(
            name="generation_llm_config",
            description="Configuration for the generation LLM",
            param_type=LLMConfig,
            available_values=[],
            default_value=DEFAULT_GENERATION_CONFIG
        )
    ]

    def __init__(self, config: DocumentRetrievalConfig) -> None:
        super().__init__(config)
        self.settings = AdditionalConfigParameter.validate_dict(
            self.ADDITIONAL_CONFIG_PARAMS, config.additional_params)
        fpm = FilePathManager()
        self.api_key_manager = SecretManager()
        working_directory = fpm.combine_paths(fpm.CACHE_DIR,
                                              "light_rag",
                                              config.config_hash)
        fpm.ensure_dir_exists(working_directory)
        logger.info(
            "Initializing LightRAG with working directory: %s", working_directory)
        llm_func, model_kwargs, rag_kwargs = self._prepare_llm(self.settings["llm_model_type"],
                                     self.settings["llm_model_name"])
        embeddings_func = self._prepare_embeddings(self.settings["llm_embedding_model_type"],
                                                   self.settings["llm_embedding_model_name"])
        
        # Initialize LightRAG
        self.rag = LightRAG(
            working_dir=working_directory,
            llm_model_func=llm_func,
            llm_model_kwargs=model_kwargs,
            embedding_func=embeddings_func,
            **(rag_kwargs or {})
        )
        self._run_indexing(config.dataset_config)

    def _run_indexing(self, dataset_config: DatasetConfig):
        dataset = DatasetManager().get_dataset(dataset_config)
        if dataset is None:
            raise ValueError("Dataset could not be loaded")
        logger.info("Indexing dataset of size: %s",
                    len(dataset.get_all_entries()))

        publications: List[Publication] = dataset.get_all_entries()
        for publication in publications[:1]:
            self.rag.insert(publication.full_text)
            
        logger.info("Finished LightRAG Indexing process")

    def retrieve(self, query_text: str) -> RetrievalAnswer:
        """
         Retrieves related knowledge entities from the knowledge base

        Args:
            query (str): The question that is used to retrieve relevant entities.
            topic_entity_id (str): The entry entity id in the graph from which the 
                search is started from.
            topic_entity_value (str): The entry entity value in the graph from which the 
                search is started from.

        Returns:
            List[Context]: A list of related knowledge entities
            Optional[str]: The answer of the retriever if they provide it.
        """
        logger.debug("---------------------------")
        logger.debug("Starting LightRAG Retrieval")
        logger.debug("Retrieving knowledge for query: %s", query_text)
        try:
            context = self.rag.query(
                query=query_text,
                param=QueryParam(
                    mode="local",
                    max_token_for_global_context=150,
                    max_token_for_local_context=150,
                    top_k=20,
                    only_need_context=True
                )
            )
            logger.debug("Retrieved context: %s", context)
        except Exception as e:
            logger.error("Error while retrieving knowledge: %s", e)
            return RetrievalAnswer(contexts=[], retriever_answer=None)

        if context is None:
            logger.debug("No context found for query: %s", query_text)
            return RetrievalAnswer(contexts=[], retriever_answer=None)

        answer = self._generate_answer(context, query_text)
        logger.debug(f"Final answer: {answer}")
        extracted_source_context = self._extract_sources(context)
        logger.debug(f"Extracted source contexts: {extracted_source_context}")

        return RetrievalAnswer(contexts=extracted_source_context,
                               retriever_answer=answer)

    def _generate_answer(self, context: str, query_text: str) -> str:
        """
        This method is copied and adapted from the original implementation of LightRAG. 

        Because currently it is not supported to return both the context and the final 
        answer, we have to use this implementation to generate the final answer from the context.
        """

        llm_config = self.settings.get("generation_llm_config")
        if not llm_config:
            raise ValueError("LLM config is missing")

        llm = LLMProvider().get_llm_adapter(llm_config)

        sys_prompt_temp = PROMPTS["rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type="Multiple Paragraphs",
            history=""
        )
        response = llm.generate(
            prompt="System: " + sys_prompt + "\nUser: " + query_text,
        )
        response = response.content
        if isinstance(response, str) and len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query_text, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        return response

    def _prepare_llm(self,
                     llm_model_type: str,
                     llm_model_name: str) -> callable:
        
        if llm_model_type == "openai":

            # Define the LLM function according to the implementation requirements
            # from LightRAG
            async def gpt_4o_mini_complete(
                prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
            ) -> str:
                keyword_extraction = kwargs.pop("keyword_extraction", None)
                if keyword_extraction:
                    kwargs["response_format"] = GPTKeywordExtractionFormat
                return await openai_complete_if_cache(
                    llm_model_name,
                    prompt,
                    api_key=self._get_api_key(llm_model_type),
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs,
                )

            return gpt_4o_mini_complete, {}, {}

        if llm_model_type == "googleapi":
            async def llm_model_func(
                prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
            ) -> str:
                return await openai_complete_if_cache(
                    "gemini-1.5-flash",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self._get_api_key(llm_model_type),
                    base_url="https://generativelanguage.googleapis.com/v1beta/",
                    **kwargs
                )

            return llm_model_func, {}, {}
        
        if llm_model_type == "ollama":
            model_kwargs = {"options": {"num_ctx": 32768}}
            if "deepseek".lower() in llm_model_name.lower():
                model_kwargs["reasoning_tag"] = "think"
                
            rag_kwargs = {
                "llm_model_name": llm_model_name,
                "llm_model_max_token_size": 32768
            }
                
            return ollama_model_complete, model_kwargs, rag_kwargs

        raise ValueError(f"Unsupported LLM model type: {llm_model_type}")

    def _prepare_embeddings(self,
                            emb_model_type: str,
                            emb_model_name: str):

        if emb_model_type == "openai":
            api_key = self._get_api_key(emb_model_type)

            @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
            async def custom_openai_embedding(
                texts: list[str],
                model: str = emb_model_name,
                base_url: str = None
            ) -> np.ndarray:
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                openai_async_client = (
                    AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
                )
                response = await openai_async_client.embeddings.create(
                    model=model, input=texts, encoding_format="float"
                )
                return np.array([dp.embedding for dp in response.data])

            return custom_openai_embedding

        if emb_model_type == "googleapi":
            api_key = self._get_api_key(emb_model_type)

            class GoogleAIEmbeddingAdapter:
                def __init__(self, func):
                    self.func = func
                    self.embedding_dim = 768  # Dimension for text-embedding-004

                async def __call__(self, texts: list[str]) -> np.ndarray:
                    return await self.func(
                        texts,
                        model=emb_model_name,
                        api_key=api_key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/"
                    )
            embedding_wrapper = GoogleAIEmbeddingAdapter(openai_embed)
            return embedding_wrapper
        
        if emb_model_type == "ollama":
            return EmbeddingFunc(
                embedding_dim=768,
                max_token_size=81972,
                func=lambda texts: ollama_embedding(texts, embed_model=emb_model_name)
            )
            
    def _extract_sources(self, answer: str) -> List[Context]:
        match = re.search(r"-----Sources-----\s*```csv(.*?)```", answer, re.DOTALL)
        if match:
            csv_content = match.group(1).strip()
            reader = csv.DictReader(io.StringIO(csv_content))
            sources = [row["content"] for row in reader]
            
            # Convert to Context objects
            sources = [Context(text=source, context_type=ContextType.DOC) for source in sources]
            return sources
        
        logger.warning("Sources block not found in the provided text.")
        return []

    def _get_api_key(self, llm_model_type: str) -> str:
        endpoint = None
        if llm_model_type == "openai":
            endpoint = EndpointType.OPENAI
        elif llm_model_type == "googleapi":
            endpoint = EndpointType.GOOGLEAI

        if endpoint is None:
            raise ValueError(f"Unsupported LLM model type: {llm_model_type}")

        try:
            return self.api_key_manager.get_api_key(endpoint)
        except Exception:
            raise ValueError(f"API key for {endpoint} is missing")
