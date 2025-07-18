# standard imports
from copy import deepcopy
from typing import Any, List, Tuple, Union
from typer.colors import BRIGHT_BLUE
from typing_extensions import override

# local imports
from sqa_system.core.config.config_manager.factory.config_manager_factory import ConfigManagerFactory
from sqa_system.core.data.models.parameter_range import ParameterRange
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.core.logging.logging import get_logger
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.models import (
    ExperimentConfig,
    Config,
    PipelineConfig
)
from sqa_system.core.config.config_manager import (
    PipelineConfigManager,
    QADatasetConfigManager,
    ExperimentConfigManager,
    EvaluatorConfigManager
)
from .pipeline_config_command_handler import PipelineConfigCommandHandler
from .qa_dataset_config_command_handler import QADatasetConfigCommandHandler
logger = get_logger(__name__)


class ExperimentConfigCommandHandler(ConfigCommandHandler[ExperimentConfig]):
    """Handles commands related to the evaluation configuration."""

    def __init__(self):
        super().__init__(ExperimentConfigManager(), "Experiment")
        self.pipeline_config_manager = PipelineConfigManager()
        self.qa_dataset_config_manager = QADatasetConfigManager()
        self.eval_config_manager = EvaluatorConfigManager()
        self.pipeline_config_manager.load_default_configs()
        self.qa_dataset_config_manager.load_default_configs()
        self.eval_config_manager.load_default_configs()
        self.config_manager.load_default_configs()

    @override
    def add_config(self):

        name = CLICommandHelper.get_text_input(
            message="Enter a name for the experiment config",
            default="",
            instruction="Leave empty for automatic name generation"
        )

        pipeline_config = CLICommandHelper.get_selection(
            "Select a base pipeline for the experiment:",
            [Choice(config, config.name)
             for config in self.pipeline_config_manager.get_all_configs()] +
            [Choice(None, "Add new pipeline")]
        )
        if pipeline_config is None:
            pipeline_config = PipelineConfigCommandHandler().add_config()
            if pipeline_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(pipeline_config):
            return None
        evaluators = CLICommandHelper.get_multiple_selection(
            "Select evaluators:",
            [Choice(evaluator, evaluator.name)
             for evaluator in self.eval_config_manager.get_all_configs()]
        )
        if CLICommandHelper.is_exit_selection(evaluators):
            return None

        qa_dataset_config = CLICommandHelper.get_selection(
            "Select the QA dataset to evaluate:",
            [Choice(config, config.name)
             for config in self.qa_dataset_config_manager.get_all_configs()] +
            [Choice(None, "Add new dataset")]
        )
        if qa_dataset_config is None:
            qa_dataset_config = QADatasetConfigCommandHandler().add_config()
            if qa_dataset_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(qa_dataset_config):
            return None
        param_ranges = self._get_param_ranges(pipeline_config)
        if CLICommandHelper.is_exit_selection(param_ranges):
            return None

        config = ConfigFactory.create_experiment_config(
            base_pipeline_config=pipeline_config,
            evaluators=evaluators,
            dataset_config=qa_dataset_config,
            param_ranges=param_ranges,
            name=name
        )
        self.config_manager.add_config(config)
        logger.info("Added evaluation configuration %s", config.name)

        if CLICommandHelper.get_confirmation("Add this " + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        return config

    def _get_param_ranges(self, config: PipelineConfig):
        """
        To allow hyperparameter tuning, the framework implements a
        ParameterRange class that allows to store ranges of parameters
        to be tuned. The following method allows the user to set up such
        parameter ranges for the selected pipeline configuration.
        The ExperimentRunner class will then use these parameter ranges
        to setup different pipelines with different configurations based
        on the parameter ranges.
        """
        all_configs_dict = config.gather_all_configs()
        configs_to_tune = CLICommandHelper.get_multiple_selection(
            "Select which configurations should be to tuned:",
            [Choice(config, config_name)
             for config_name, config in all_configs_dict.items()]
        )
        if CLICommandHelper.is_exit_selection(configs_to_tune):
            return configs_to_tune

        if len(configs_to_tune) == 0:
            return []

        param_ranges = []
        for config_to_tune in configs_to_tune:
            config_name = config_to_tune.name
            model_fields = config_to_tune.to_dict()
            CLICommandHelper.print(
                f"Tuning {config_to_tune.name}", color=BRIGHT_BLUE)
            params_to_tune = CLICommandHelper.get_multiple_selection(
                "Select which parameters should be tuned:",
                [Choice(model_field_key,
                        f"{model_field_key} ({getattr(config_to_tune, model_field_key)})")
                 for model_field_key, _ in model_fields.items()]
            )
            if len(params_to_tune) == 0:
                continue
            for param_to_tune in params_to_tune:
                new_values = self._get_new_value_based_on_type(
                    config_to_tune, param_to_tune
                )
                if new_values is None or len(new_values) == 0:
                    continue

                # For list type of parameters, the handler returns a special structure that
                # allows to identify the index of the list element that should be changed.
                # Here we extract the index from this structure if it exists.
                index = -1
                if isinstance(new_values, list) and isinstance(new_values[0], Tuple):
                    index = new_values[0][0]
                    new_values = new_values[0][1]

                if index >= 0:
                    param_ranges.append(
                        ParameterRange(
                            config_name=config_name,
                            parameter_name=param_to_tune,
                            values=new_values,
                            list_index=index
                        )
                    )
                else:
                    param_ranges.append(
                        ParameterRange(
                            config_name=config_name,
                            parameter_name=param_to_tune,
                            values=new_values
                        )
                    )

        return param_ranges

    def _get_new_value_based_on_type(self,
                                     content_object: Any,
                                     param_to_tune: str) -> Union[List[Any], None]:
        """
        This function is used to get a new value based on the type of the field.
        """
        if isinstance(content_object, dict):
            field_value = content_object.get(param_to_tune)
            if not field_value:
                CLICommandHelper.print_error(
                    f"The selected parameter cannot be tuned using this UI: {param_to_tune}"
                )
                return None
        else:
            field_value = getattr(content_object, param_to_tune)
        # These are the types that are supported for tuning
        # For each of the types we have a corresponding handler
        # function.
        type_handlers = {
            int: self._input_primitive_type,
            float: self._input_primitive_type,
            bool: self._input_primitive_type,
            str: self._input_primitive_type,
            Config: self._input_config_type,
            dict: self._input_config_type,
            list: self._input_list_type
        }

        handler = type_handlers.get(type(field_value))

        if handler:
            return handler(field_value, param_to_tune)

        CLICommandHelper.print_error(
            f"The selected parameter cannot be \
                tuned using this UI: {type(field_value).__name__}"
        )
        return None

    def _input_primitive_type(self,
                               field_value: Union[int, float, bool, str],
                               param_to_tune: str) -> Union[List[Any], None]:
        """
        Allows the user to input new values for primitive types.
        """
        type_name = type(field_value).__name__
        new_values = CLICommandHelper.get_text_input(
            f"Enter tuning values for {param_to_tune} ({type_name}): ",
            instruction="(Enter multiple values separated by comma)"
        )
        if CLICommandHelper.is_exit_selection(new_values):
            return None
        return self._convert_values_string_to_type_list(new_values, type(field_value))

    def _input_config_type(self,
                            field_value: Config,
                            param_to_tune: str) -> Union[List[Config], None]:
        """
        Configurations are special types. Through the ConfigManagerFactory we the
        corresponding ConfigManager for the given configuration. This allows us to
        select other configurations of the same type that should be used for tuning.
        """
        config_manager = ConfigManagerFactory.get_config_manager(field_value)
        config_manager.load_default_configs()
        selected_configs = CLICommandHelper.get_multiple_selection(
            f"Select configurations for {param_to_tune} to compare:",
            [Choice(config, config.name)
             for config in config_manager.get_all_configs()]
        )
        if CLICommandHelper.is_exit_selection(selected_configs):
            return None
        return selected_configs

    def _input_dict_type(self,
                          field_value: dict,
                          param_to_tune: str,
                          depth: int = 0) -> Union[List[dict], None]:
        """
        Dictionaries are the most complex types to handle because it can
        have further nested dictionaries or configurations. The following
        method allows the user to select which parameters should be tuned
        for the given dictionary.
        """
        if depth > 5:
            CLICommandHelper.print_error(
                f"Maximum nesting depth reached for parameter {param_to_tune}")
            return None

        result = [deepcopy(field_value)]

        keys_to_tune = CLICommandHelper.get_multiple_selection(
            f"Select which parameters should be tuned for {param_to_tune}:",
            [Choice(key, key) for key in field_value.keys()]
        )

        for key in keys_to_tune:
            sub_values = self._get_new_value_based_on_type(field_value, key)

            if CLICommandHelper.is_exit_selection(sub_values):
                return None

            if sub_values:
                new_result = []
                for original in result:
                    for sub_value in sub_values:
                        new_dict = deepcopy(original)
                        self._set_nested_dict_value(
                            new_dict, key.split('.'), sub_value)
                        new_result.append(new_dict)
                result = new_result

        return result if len(result) > 1 else None

    def _set_nested_dict_value(self, d: dict, keys: List[str], value: Any):
        """Helper method to set a value in a nested dictionary."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _input_list_type(self, field_value: list, param_to_tune: str) -> Union[List[Any], None]:
        """
        Handles tuning for list-type parameters with varying entry types.
        """

        # If the list is empty, we currently don't allow further configuration
        # because the type of entries in the list is unknown.
        if not field_value:
            CLICommandHelper.print_error(
                f"The selected parameter is an empty list: {param_to_tune}"
            )
            return None

        # Determine the type of the first entry in the list
        entry_type = type(field_value[0])

        if entry_type in (int, float, bool, str):
            # If the entries are of primitive types, we ask the user to input
            # a complete new list of values
            new_values_input = CLICommandHelper.get_text_input(
                f"Enter new values for {param_to_tune} (current values: {field_value}):",
                instruction="(Enter multiple values separated by comma)"
            )

            if CLICommandHelper.is_exit_selection(new_values_input):
                return None

            return self._convert_values_string_to_type_list(new_values_input, entry_type)

        if issubclass(entry_type, Config):
            # In the case of a config type, the user is askes to provide a index of the
            # config that should be changed. He can then select from the list of available
            # configurations in the current config manager to exchange the selected config.
            for index, config in enumerate(field_value):
                CLICommandHelper.print(
                    f"[{index}] {config.name}",
                    color=BRIGHT_BLUE
                )

            config_to_change = CLICommandHelper.get_selection(
                "Select the configuration you want to tune:",
                [Choice(config, config.name) for config in field_value]
            )
            index_of_config_to_change = field_value.index(config_to_change)

            if CLICommandHelper.is_exit_selection(config_to_change):
                return None

            # Here we use the configuration manager of that specificly selected config
            # which stores all other configs of that type, to allow the user to select
            # other configurations that should be used for tuning.
            config_manager = ConfigManagerFactory.get_config_manager(
                config_to_change)
            all_configs = config_manager.get_all_configs()
            selected_configs = CLICommandHelper.get_multiple_selection(
                "Select the configurations that should be used for tuning:",
                [Choice(config, config.name) for config in all_configs]
            )

            if selected_configs is None or CLICommandHelper.is_exit_selection(selected_configs):
                return None

            return [(index_of_config_to_change, selected_configs)]

        CLICommandHelper.print_error(
            f"The selected parameter type is not supported for tuning: {entry_type.__name__}"
        )
        return None

    def _convert_values_string_to_type_list(self, values_str: str, value_type: type) -> List[Any]:
        """
        Converts a comma-separated string of values to a list of the specified type.
        """
        if not values_str:
            raise ValueError("No values provided")

        values = [value.strip()
                  for value in values_str.split(",") if value.strip()]
        if not values:
            raise ValueError("No valid values provided")

        result = []
        for value in values:
            try:
                typed_value = value_type(value)
                result.append(typed_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid value: {value} (not a {value_type.__name__})") from exc

        return result
