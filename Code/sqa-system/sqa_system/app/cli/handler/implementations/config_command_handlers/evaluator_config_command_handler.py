from typing_extensions import override
from sqa_system.core.config.models import EvaluatorConfig
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.config.config_manager import EvaluatorConfigManager
from sqa_system.experimentation.evaluation.factory.evaluator_factory import EvaluatorType, EvaluatorFactory


class EvaluatorConfigCommandHandler(ConfigCommandHandler[EvaluatorConfig]):
    """Class responsible for handling Evaluator configurations"""

    def __init__(self):
        super().__init__(EvaluatorConfigManager(), "Evaluator")
        self.config_manager.load_default_configs()

    @override
    def add_config(self):
        while True:
            evaluator_name = CLICommandHelper.get_text_input(
                message="Enter a name for the Evaluator config",
                default="",
            )
            if not evaluator_name:
                CLICommandHelper.print("Name cannot be empty.")
                continue
            break
        evaluator_type = CLICommandHelper.get_selection(
            "Select Evaluator:",
            [Choice(eval_type.value, eval_type.value)
             for eval_type in EvaluatorType],
        )
        if CLICommandHelper.is_exit_selection(evaluator_type):
            return None

        evaluator_class = EvaluatorFactory.get_evaluator_class(evaluator_type)
        additional_config_params = self._get_additional_config_params(
            evaluator_class.ADDITIONAL_CONFIG_PARAMS)

        if not additional_config_params:
            config = ConfigFactory.create_evaluator_config(
                name=evaluator_name,
                evaluator_type=evaluator_type,
                **additional_config_params)
        else:
            config = ConfigFactory.create_evaluator_config(
                name=evaluator_name,
                evaluator_type=evaluator_type,
                **additional_config_params)

        self.config_manager.add_config(config)

        if CLICommandHelper.get_confirmation("Add this " + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config
