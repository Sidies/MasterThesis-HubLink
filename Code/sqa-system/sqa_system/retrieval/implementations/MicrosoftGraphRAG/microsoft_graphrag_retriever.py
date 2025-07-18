import os
import subprocess
import yaml
from typing import List, ClassVar
import weave


from sqa_system.core.data.secret_manager import SecretManager, EndpointType
from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.core.config.models import DocumentRetrievalConfig
from sqa_system.core.data.dataset_manager import DatasetManager
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger

from ...base.document_retriever import DocumentRetriever

logger = get_logger(__name__)


class GraphRAGRetriever(DocumentRetriever):

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="query_method",
            description=("The query method for GraphRAG retrieval which can be local "
                         "or global."),
            param_type=str,
            available_values=["local", "global"],
            default_value="local"
        ),
        AdditionalConfigParameter(
            name="openai_model_name",
            description=(
                "The name of the OpenAI model to use for GraphRAG retrieval."),
            param_type=str,
            available_values=[],
            default_value="gpt-4o-mini"
        ),
        AdditionalConfigParameter(
            name="openai_embedding_model_name",
            description=("The name of the OpenAI embedding model to use."),
            param_type=str,
            available_values=[],
            default_value="text-embedding-3-small"
        ),
        AdditionalConfigParameter(
            name="chunk_size",
            description=("The chunk size for GraphRAG retrieval."),
            param_type=int,
            available_values=[],
            default_value=1200
        ),
        AdditionalConfigParameter(
            name="chunk_overlap",
            description=("The chunk overlap for GraphRAG retrieval."),
            param_type=int,
            available_values=[],
            default_value=100
        ),
        AdditionalConfigParameter(
            name="max_cluster_size",
            description=("The maximum cluster size for GraphRAG retrieval."),
            param_type=int,
            available_values=[],
            default_value=10
        ),
        AdditionalConfigParameter(
            name="embed_graph",
            description=("Whether to embed the graph for GraphRAG retrieval."),
            param_type=bool,
            available_values=[],
            default_value=False,
        )
    ]

    def __init__(self, config: DocumentRetrievalConfig) -> None:
        super().__init__(config)

        self.settings = AdditionalConfigParameter.validate_dict(
            self.ADDITIONAL_CONFIG_PARAMS, config.additional_params
        )
        self.fpm = FilePathManager()

        self._prepare_working_directory()
        self._prepare_api_key()
        self._update_settings()
        self._run_indexing()

    def retrieve(self, query_text: str) -> RetrievalAnswer:
        logger.info("Starting GraphRAG retrieval for query: %s", query_text)
        try:
            cmd = [
                "graphrag", "query",
                "--root", self.workspace,
                "--method", self.settings["query_method"],
                "--query", query_text
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            logger.debug("GraphRAG query output: %s", output)
        except Exception as e:
            logger.error("Error during GraphRAG retrieval: %s", e)
            return RetrievalAnswer(contexts=[], retriever_answer=None)
        
        answer = self._extract_answer(output)
        return RetrievalAnswer(contexts=[], retriever_answer=answer )
    
    def _extract_answer(self, output: str) -> str:
        """
        Extracts the final answer from the GraphRAG output.
        It looks for the marker 'SUCCESS: Local Search Response:' and returns everything after it.
        If the marker is not found, returns the original output.
        """
        index = -1
        marker = ""
        if "SUCCESS: Local Search Response:" in output:
            index = output.find("SUCCESS: Local Search Response:")
            marker = "SUCCESS: Local Search Response:"
        elif "SUCCESS: Global Search Response:" in output:
            index = output.find("SUCCESS: Global Search Response:")
            marker = "SUCCESS: Global Search Response:"
        else:
             return "No answer has been generated."

        answer = output[index + len(marker):].strip()
        return answer

    def _prepare_working_directory(self):
        self.workspace = self.fpm.combine_paths(
            self.fpm.CACHE_DIR, "microsoft_graphrag", self.config.dataset_config.config_hash)
        self.fpm.ensure_dir_exists(self.workspace)

        # Initialize the GraphRAG workspace if not already initialized
        if not os.path.exists(os.path.join(self.workspace, ".env")):
            logger.info(
                "GraphRAG workspace not initialized. Running 'graphrag init'.")
            subprocess.run(["graphrag", "init", "--root",
                           self.workspace], check=True)

        logger.info(
            f"GraphRAG working directory has been initialized to: {self.workspace}")

    def _prepare_api_key(self):
        self.api_key = SecretManager().get_api_key(EndpointType.OPENAI)
        env_path = os.path.join(self.workspace, ".env")
        key_line = f"GRAPHRAG_API_KEY={self.api_key}\n"

        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        updated = False
        for i, line in enumerate(lines):
            if line.startswith("GRAPHRAG_API_KEY="):
                lines[i] = key_line
                updated = True
                break
        if not updated:
            lines.append(key_line)
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.debug("GraphRAG API key is prepared.")

    def _update_settings(self):
        settings_path = os.path.join(self.workspace, "settings.yaml")
        if not os.path.exists(settings_path):
            raise ValueError("Settings file does not exist")

        with open(settings_path, "r", encoding="utf-8") as f:
            try:
                settings_data = yaml.safe_load(f)
            except Exception as e:
                logger.error("Error loading settings.yaml: %s", e)
                return

        settings_data["llm"]["model"] = self.settings["openai_model_name"]
        settings_data["embeddings"]["llm"]["model"] = self.settings["openai_embedding_model_name"]
        settings_data["chunks"]["size"] = self.settings["chunk_size"]
        settings_data["chunks"]["overlap"] = self.settings["chunk_overlap"]
        settings_data["cluster_graph"]["max_cluster_size"] = self.settings["max_cluster_size"]
        settings_data["embed_graph"]["enabled"] = self.settings["embed_graph"]

        with open(settings_path, "w", encoding="utf-8") as f:
            yaml.dump(settings_data, f)
        logger.debug("Updated settings.yaml")

    @weave.op(name="Microsoft GraphRAG Indexing")
    def _run_indexing(self):
        """
        Indexes the dataset by writing the data into the input folder of the working directory.
        """
        dataset = DatasetManager().get_dataset(self.config.dataset_config)
        if dataset is None:
            raise ValueError("Dataset could not be loaded")

        publications: List[Publication] = dataset.get_all_entries()
        logger.info("Indexing dataset of size: %s", len(publications))
        input_dir = os.path.join(self.workspace, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Check if context files already exist
        if os.path.exists(os.path.join(self.workspace, "cache")):
            logger.info("GraphRAG cache already exists. Skipping indexing.")
            return

        # Because graph rag expects the data as text files, we now write each publication
        # to a text file in the input folder
        for publication in publications:
            safe_doi = publication.doi.replace("/", "_")
            with open(os.path.join(input_dir, f"{safe_doi}.txt"), "w", encoding="utf-8") as f:
                f.write(str(publication))
            logger.debug("Wrote publication to file: %s", safe_doi)
            
        logger.info("Starting GraphRAG indexing...")
        subprocess.run(["graphrag", "index", "--root", self.workspace], check=True)
        logger.info("GraphRAG indexing completed.")
