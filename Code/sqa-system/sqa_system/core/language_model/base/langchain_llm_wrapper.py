import re
from typing import Any, Dict, Iterator, List, Optional, Union
from typing_extensions import override
import weave

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_community.callbacks.openai_info import (
    MODEL_COST_PER_1K_TOKENS,
    standardize_model_name,
    get_openai_token_cost_for_model,
    TokenType
)
from pydantic import BaseModel

from sqa_system.core.logging.logging import get_logger

from ..llm_stat_tracker import LLMStatTracker, LLMStats

logger = get_logger(__name__)


class LangchainLLMWrapper(BaseChatModel):
    """
    Custom wrapper function for Langchain LLMs.
    We use this to track the tokens used by the LLM and make sure that special tokens
    are not returned in the output.

    Implementation based on https://python.langchain.com/docs/how_to/custom_chat_model/
    """
    base_llm: BaseChatModel

    @override
    def with_structured_output(
        self,
        schema: Union[Dict, type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        We overwrite this to not crash the system when the LLM does not support
        structured output. We just return the default output in that case.
        """
        try:
            return self.base_llm.with_structured_output(schema, include_raw=include_raw, **kwargs)
        except NotImplementedError:
            logger.warning(
                "Structured output is not supported by the LLM falling back to default output.")
        return self

    @override
    @weave.op()
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The main generate method that we overwrite to add our custom logic.
        Here we add the logic to track the tokens used by the LLM and make sure that
        special tokens are not returned in the output.
        """

        logger.debug(f"Asking LLM: {self._llm_type}")

        # pylint: disable=protected-access
        response = self.base_llm._generate(
            messages, stop, run_manager, **kwargs)

        if not response:
            logger.error("No response from LLM")
            return ChatResult()

        if len(response.generations) >= 1:
            pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

            # Clean the text field
            cleaned_text = pattern.sub("", response.generations[0].text)
            response.generations[0].text = cleaned_text

            # Clean the message content field, if it exists
            if hasattr(response.generations[0], "message") and hasattr(response.generations[0].message, "content"):
                cleaned_content = pattern.sub(
                    "", response.generations[0].message.content)
                response.generations[0].message.content = cleaned_content

        else:
            logger.warning("No generations found in response")

        self._add_llm_stats(response)

        return response

    @override
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError("Streaming is not supported.")

    @override
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        # pylint: disable=protected-access
        return self.base_llm._llm_type

    @override
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes, making it possible to monitor LLMs.
        """
        if hasattr(self.base_llm, "model_name"):
            return {
                "model_name": self.base_llm.model_name
            }
        if hasattr(self.base_llm, "model"):
            return {
                "model_name": self.base_llm.model
            }
        return {
            # pylint: disable=protected-access
            "model_name": self.base_llm._llm_type
        }

    def _add_llm_stats(self, response: ChatResult):
        """
        Adds tracking information about the LLM response object.

        Args:
            response: The response object from the LLM.
        """
        if response.llm_output is not None and response.llm_output.get("token_usage", None):
            self._add_llm_data_from_output_tokens(response)
        elif response.generations and hasattr(response.generations[0].message, "usage_metadata"):
            usage_metadata = response.generations[0].message.usage_metadata
            self._add_llm_data_from_usage_metadata(usage_metadata)

    def _add_llm_data_from_output_tokens(self, response: ChatResult):
        """
        Helper function to add tracking data from the LLM response object.

        Args:
            response: The response object from the LLM.
        """
        try:
            total_tokens = response.llm_output["token_usage"]["total_tokens"]
            completion_tokens = response.llm_output["token_usage"]["completion_tokens"]
            prompt_tokens = response.llm_output["token_usage"]["prompt_tokens"]
            if "cache_read" in response.llm_output["token_usage"].get("input_token_details", {}):
                prompt_tokens_cached = response.llm_output["token_usage"]["input_token_details"][
                    "cache_read"
                ]
            else:
                prompt_tokens_cached = 0

            model_name = standardize_model_name(
                response.llm_output.get("model_name", "")
            )

            if model_name in MODEL_COST_PER_1K_TOKENS:
                uncached_prompt_tokens = prompt_tokens - prompt_tokens_cached
                uncached_prompt_cost = get_openai_token_cost_for_model(
                    model_name, uncached_prompt_tokens, token_type=TokenType.PROMPT
                )
                cached_prompt_cost = get_openai_token_cost_for_model(
                    model_name, prompt_tokens_cached, token_type=TokenType.PROMPT_CACHED
                )
                prompt_cost = uncached_prompt_cost + cached_prompt_cost
                completion_cost = get_openai_token_cost_for_model(
                    model_name, completion_tokens, token_type=TokenType.COMPLETION
                )
            else:
                completion_cost = 0
                prompt_cost = 0

            LLMStatTracker().add_stats(stats=LLMStats(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost=completion_cost + prompt_cost,
            ))
        except KeyError as e:
            logger.warning(f"Token usage information missing: {e}.")
            logger.debug(response.llm_output)

    def _add_llm_data_from_usage_metadata(self, usage_metadata: dict) -> None:
        """
        Alternative helper method to add tracking data from the LLM response object.
        """
        try:
            total_tokens = usage_metadata["total_tokens"]
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]

            LLMStatTracker().add_stats(stats=LLMStats(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                cost=0
            ))
        except KeyError as e:
            logger.warning(f"Token usage information missing: {e}.")
            logger.debug(usage_metadata)
