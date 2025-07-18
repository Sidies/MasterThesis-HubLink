from typing import Type, List
from enum import Enum
from typing_extensions import override
from orkg import ORKG, Hosts
import time

from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.config.models import AdditionalConfigParameter, RestrictionType
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.config.models import KnowledgeGraphConfig
from sqa_system.core.data.models import PublicationDataset
from sqa_system.core.data.secret_manager import SecretManager, SecretType
from sqa_system.core.logging.logging import get_logger

from ....implementations.orkg_remote_graph import ORKGRemoteGraph
from ....factory.base.knowledge_graph_builder import KnowledgeGraphBuilder
from .orkg_template_builder.orkg_template_builder import ORKGTemplateBuilder

from .orkg_template_builder.implementations.paper_class_block import PaperClassBlock
from .orkg_template_builder.implementations.paper_class_block_flattened import PaperClassFlattenedBlock
from .orkg_template_builder.implementations.research_level_block import ResearchLevelBlock
from .orkg_template_builder.implementations.research_objects_block import (
    FirstResearchObjectBlock, SecondResearchObjectBlock, AllResearchObjectsBlock)
from .orkg_template_builder.implementations.research_objects_flattened import (
    FirstResearchObjectFlattenedBlock, SecondResearchObjectFlattenedBlock, AllResearchObjectsFlattenedBlock)
from .orkg_template_builder.implementations.research_level_block_flattened import ResearchLevelFlattenedBlock
from .orkg_template_builder.implementations.validity_block import ValidityBlock
from .orkg_template_builder.implementations.validity_block_flattened import ValidityFlattenedBlock
from .orkg_template_builder.implementations.evidence_block import EvidenceBlock
from .orkg_template_builder.implementations.evidence_block_flattened import EvidenceFlattenedBlock
from .orkg_template_builder.contribution import Contribution

logger = get_logger(__name__)


class FulltextBuildingBlock(Enum):
    """
    Enum class for building blocks related to the fulltext of a publication.
    """
    
class MetadataBuildingBlock(Enum):
    """
    Enum class for building blocks that add metadata to the publication.
    """


class AnnotationBuildingBlock(Enum):
    """
    Enum class for building blocks related to the expert annotations of
    a publication.
    """
    PAPER_CLASS = PaperClassBlock
    RESEARCH_LEVEL = ResearchLevelBlock
    ALL_RESEARCH_OBJECTS = AllResearchObjectsBlock
    FIRST_RESEARCH_OBJECT = FirstResearchObjectBlock
    SECOND_RESEARCH_OBJECT = SecondResearchObjectBlock
    VALIDITY = ValidityBlock
    FIRST_RESEARCH_OBJECT_FLATTENED = FirstResearchObjectFlattenedBlock
    SECOND_RESEARCH_OBJECT_FLATTENED = SecondResearchObjectFlattenedBlock
    ALL_RESEARCH_OBJECTS_FLATTENED = AllResearchObjectsFlattenedBlock
    RESEARCH_LEVEL_FLATTENED = ResearchLevelFlattenedBlock
    PAPER_CLASS_FLATTENED = PaperClassFlattenedBlock
    VALIDITY_FLATTENED = ValidityFlattenedBlock
    EVIDENCE = EvidenceBlock
    EVIDENCE_FLATTENED = EvidenceFlattenedBlock


class ORKGKnowledgeGraphFactory(KnowledgeGraphBuilder):
    """
    Factory class responsible for establishing a connection with ORKG
    and adding the publication data to the ORKG graph.
    """

    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="contribution_building_blocks",
            description=("The building blocks to use for the graph creation"),
            param_type=dict[str, List[str]],
            default_value={"Publication Overview": [
                block.value for block in AnnotationBuildingBlock]},
        ),
        AdditionalConfigParameter(
            name="force_cache_update",
            description=("Whether the ORKG cache should be updated to the newest data "
                         "in the graph."),
            param_type=bool,
            default_value=False
        ),
        AdditionalConfigParameter(
            name="force_publication_update",
            description=("Whether the publication data should be updated in the graph "
                         "even if it already exists."),
            param_type=bool,
            default_value=False
        ),
        AdditionalConfigParameter(
            name="subgraph_root_entity_id",
            description=("For our purpose, we do not want to query the whole ORKG graph. Some retrievers have "
                         "an indexing process which would be costly, if this is done on the whole graph. "
                         "Therefore, we will limit the graph to only a subgraph. The id of this subgraph "
                         "is defined in this parameter."),
            param_type=str,
            default_value="R659055"

        ),
        AdditionalConfigParameter(
            name="orkg_base_url",
            description=("The Base URL of the ORKG instance to connect to."),
            param_type=str,
            default_value="https://sandbox.orkg.org"
        ),
        AdditionalConfigParameter(
            name="limit_publications",
            description=("The maximum number of valid publications from the graph that is allowed. This "
                         "option is partifularly useful for debugging to avoid caching the whole data "
                         "and not having to delete data from the graph"),
            param_type=int,
            default_value=-1,
            param_restriction=RestrictionType.GREQ_THAN_MINUS_1,
        )
    ]

    def __init__(self):
        self.orkg = None
        self.contribution_cache = {}
        self._init_orkg()

    def _init_orkg(self):
        """
        Initializes the connection to the ORKG api using the credentials
        stored in the secret manager.
        """
        secret_manager = SecretManager()
        try:
            email = secret_manager.get_secret(SecretType.EMAIL, "orkg_mail")
        except Exception as exc:
            logger.error("The email for ORKG was not found in the secret manager. " +
                         "Please add it to the secret manager through the main menu, " +
                         "by adding the email with the key 'orkg_mail'.")
            raise ValueError(
                "The email for ORKG was not found in the secret manager. " +
                "Please add it to the secret manager through the main menu, " +
                "by adding the email with the key 'orkg_mail'.") from exc
        try:
            password = secret_manager.get_secret(
                SecretType.PWD, "orkg_password")
        except Exception as exc:
            logger.error("The password for ORKG was not found in the secret manager. "
                         "Please add it to the secret manager through the main menu, "
                         "by adding the password with the key 'orkg_password'.")
            raise ValueError(
                "The password for ORKG was not found in the secret manager. "
                "Please add it to the secret manager through the main menu, "
                "by adding the password with the key 'orkg_password'.") from exc

        creds = (email, password)
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                self.orkg = ORKG(host=Hosts.SANDBOX, creds=creds)
                logger.info("Connected to the ORKG API.")
                break
            except Exception as exc:
                logger.error("ORKG connection attempt %d/%d failed: %s",
                             attempt, max_retries, exc)
                if attempt < max_retries:
                    sleep_time = 2 ** (attempt - 1)
                    time.sleep(sleep_time)
                else:
                    raise ValueError("Could not connect to the ORKG API after "
                                     f"{max_retries} attempts.") from exc

    @classmethod
    @override
    def get_knowledge_graph_class(cls) -> Type[ORKGRemoteGraph]:
        return ORKGRemoteGraph

    @override
    def _create_knowledge_graph(self,
                                publications: PublicationDataset,
                                config: KnowledgeGraphConfig) -> ORKGRemoteGraph:
        """
        The main function for populating the ORKG graph with the publication data.
        It connects to the ORKG API and adds the publication data to the ORKG graph.
        It uses the ORKGTemplateBuilder to build the template for the publication
        and the contributions.

        Args:
            publications (PublicationDataset): The dataset of publications to add
                to the ORKG graph.
            config (KnowledgeGraphConfig): The configuration for the ORKG graph
                creation. 

        Returns:
            ORKGRemoteGraph: The ORKG graph connection object. When returned, the
                graph is already populated with the publication data.
        """
        settings = AdditionalConfigParameter.validate_dict(
            self.ADDITIONAL_CONFIG_PARAMS, config.additional_params)
        contributions = settings.get(
            "contribution_building_blocks", {"Publication Overview": None})
        contribution_names = contributions.keys()
        force_publication_update = settings.get(
            "force_publication_update", False)

        # Connect to the ORKG
        graph = ORKGRemoteGraph(config)

        # Main Loop
        ProgressHandler().add_task(
            string_id="orkg_publication_update",
            description="Checking ORKG Publications..",
            total=len(publications.data),
            reset=True
        )
        need_caching_update = False
        for _, publication in publications.iterate_entries():
            ProgressHandler().update_task_by_string_id("orkg_publication_update", 1)

            try:
                has_all_contributions = self._check_if_all_contributions_match(
                    publication, contribution_names)
                if (not force_publication_update and has_all_contributions):
                    logger.debug(
                        "The publication: '%s' and the contributions: '%s' already exist in the ORKG graph.",
                        publication.title, contribution_names)
                    continue

                if force_publication_update:
                    self._delete_publication_from_graph(publication)

                builder = self._prepare_builder(config, publication)
                data_dict = builder.create_template_for_publication(
                    publication=publication,
                    research_field_id=config.additional_params.get(
                        "subgraph_root_entity_id", "R659055")
                )
                try:
                    result = self._retry_orkg_call(self.orkg.papers.add,
                                                   params=data_dict,
                                                   merge_if_exists=True)
                    if result.succeeded:
                        logger.debug(
                            "The paper: '%s' was successfully added or updated.",
                            publication.title)
                    else:
                        logger.error(
                            "There was an error adding the paper %s to the ORKG graph: %s",
                            publication.title, result.content)
                        raise ValueError(
                            f"There was an error adding the paper {publication.title} to the ORKG graph: {result.content}")

                except ConnectionError:
                    logger.error(
                        "There was an error connecting to the ORKG API. Retrying...")
                    self._init_orkg()
                    continue
                except Exception as e:
                    logger.error(
                        "There was an error adding the paper %s to the ORKG graph: %s. Retrying...",
                        publication.title, e)
                    self._init_orkg()
                    continue
            except ConnectionError:
                logger.error(
                    "There was an error connecting to the ORKG API. Retrying...")
                self._init_orkg()
                continue
        ProgressHandler().finish_by_string_id("orkg_publication_update")

        # Start the caching process
        if config is not None:
            if config.additional_params.get("force_cache_update", False) or need_caching_update:
                graph.cache_subgraph()
            else:
                graph.update_cache_if_not_exists()
        return graph

    def _check_if_contribution_in_publication(self,
                                              publication: Publication,
                                              contribution_name: str) -> bool:
        """
        Checks if the contribution with the given name already exists for the given publication.
        This is useful to avoid adding the same contribution multiple times.

        Note: That this check is done only by name of the contribution and not its content!

        Args:
            publication (Publication): The publication to check.
            contribution_name (str): The name of the contribution to check.

        Returns:
            bool: True if the contribution exists, False otherwise.
        """
        contributions = self._get_contributions_from_publication(
            publication=publication)
        for contribution in contributions:
            if contribution["label"] == contribution_name:
                return True
        return False

    def _get_contributions_from_publication(self,
                                            publication: Publication) -> List[str]:
        """
        Returns the contributions of a publication.

        Args:
            publication (Publication): The publication to return the contributions for.

        Returns:
            List[str]: The names of the contributions of the publication which are
                currently in the ORKG graph.
        """
        # First look in the cache whether the contributions where already fetched
        contributions = self.contribution_cache.get(
            publication.doi, None)
        if contributions is None:
            # If not we need to fetch the contributions from the ORKG
            result = self._retry_orkg_call(
                self.orkg.papers.by_doi, publication.doi)
            if not result.succeeded or len(result.content) == 0:
                result = self._retry_orkg_call(
                    self.orkg.papers.by_title, publication.title)
            if not result.succeeded or result.content is None or len(result.content) == 0:
                return False
            # Sometimes the return is a dict sometimes it is a list
            if isinstance(result.content, dict) and result.content.get("content", None):
                content = result.content["content"][0]
            elif isinstance(result.content, list):
                content = result.content[0]
            else:
                content = result.content
                
            if content is None or len(content["contributions"]) == 0:
                return False
            contributions = content["contributions"]
            self.contribution_cache[publication.doi] = contributions
        return contributions

    def _check_if_all_contributions_match(self,
                                          publication: Publication,
                                          contribution_names: List[str]) -> bool:
        """
        Checks if the publication has all the contributions with the given names.

        Args:
            publication (Publication): The publication to check.
            contribution_names (List[str]): The names of the contributions to check.

        Returns:
            bool: True if the publication has all the contributions, False otherwise.
        """
        contributions = self._get_contributions_from_publication(
            publication=publication)
        for contribution_name in contribution_names:
            contribution_match = False
            for contribution in contributions:
                if contribution["label"] == contribution_name:
                    contribution_match = True
                    break
            if not contribution_match:
                return False

        return True

    def _delete_publication_from_graph(self,
                                       publication: Publication):
        """
        Deletes the publication from the ORKG graph if it already exists.

        Args:
            publication (Publication): The publication to delete from the ORKG graph.
        """
        result = self._retry_orkg_call(
            self.orkg.papers.by_doi, publication.doi)
        if not result.succeeded or result.content is None or len(result.content) == 0:
            logger.debug(
                "The publication: '%s' does not exist in the ORKG graph. No need to delete",
                publication.title)
            return
        resource_id = result.content[0]["id"]

        if self._retry_orkg_call(self.orkg.resources.exists, id=resource_id):
            self._retry_orkg_call(self.orkg.papers.delete, id=resource_id)
            if self._retry_orkg_call(self.orkg.papers.exists, id=resource_id):
                logger.debug(f"Resource with ID {resource_id} was not deleted")
            else:
                logger.debug(f"Resource with ID {resource_id} was deleted")
        else:
            logger.debug(f"Resource with ID {resource_id} does not exist.")

    def _prepare_builder(self,
                         config: KnowledgeGraphConfig,
                         publication: Publication) -> ORKGTemplateBuilder:
        """
        Prepares the template builder with the building blocks specified in the configuration.
        Skips contributions that already exist in the publication.

        Args:
            config (KnowledgeGraphConfig): The configuration for the ORKG graph creation.
            publication (Publication): The publication to add to the ORKG graph.
        Returns:
            ORKGTemplateBuilder: The ORKG template builder with the specified building blocks.
        """
        builder = ORKGTemplateBuilder(
            llm_config=config.extraction_llm,
            context_size=config.extraction_context_size
        )
        contributions_plan = config.additional_params.get(
            "contribution_building_blocks", None)

        if contributions_plan is None:
            logger.warning(
                "No contribution building blocks were specified in the configuration.")
            return builder

        for contribution_name, blocks in contributions_plan.items():
            if self._check_if_contribution_in_publication(
                    publication=publication,
                    contribution_name=contribution_name):
                continue

            contribution = Contribution(
                llm_config=config.extraction_llm,
                context_size=config.extraction_context_size,
                name=contribution_name
            )
            self._add_blocks_to_contribution(blocks, contribution)
            builder.add_contribution(contribution)
        return builder

    def _add_blocks_to_contribution(self, blocks: List[str], contribution: Contribution):
        """
        Given a list of blocks, these are added to the provided contribution.

        Args:
            blocks (List[str]): The list of blocks to add to the contribution.
            contribution (Contribution): The contribution to which the blocks will be added.
        """
        for block_name in blocks:
            for block in FulltextBuildingBlock:
                if block_name.lower() == block.name.lower():
                    contribution.add_paper_content_template_block(
                        block.value())
                    break

            for block in AnnotationBuildingBlock:
                if block_name.lower() == block.name.lower():
                    contribution.add_annotation_block(
                        block.value())
                    break
                
            for block in MetadataBuildingBlock:
                if block_name.lower() == block.name.lower():
                    contribution.add_metadata_block(
                        block.value())
                    break

    def _retry_orkg_call(self, func, *args, max_retries: int = 6, **kwargs):
        """
        The orkg API often fails to respond so we needed to add a retry mechanism that wraps
        the API call and retries it a few times before failing.

        Args:
            func: The function to call.
            *args: The arguments to pass to the function.
            max_retries (int): The maximum number of retries to attempt.
        """
        retries = 0
        last_exception = None

        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"ORKG API call failed, retrying ({retries}/{max_retries}): {exc}")
                    try:
                        self._init_orkg()
                    except Exception as init_exc:
                        logger.error(
                            f"Failed to reinitialize ORKG connection: {init_exc}")
                else:
                    logger.error(
                        f"ORKG API call failed after {max_retries} retries: {exc}")

        raise last_exception
