import argparse
import weave

from sqa_system.core.language_model.errors.api_key_missing_error import APIKeyMissingError
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.command_handler import CommandHandler
from sqa_system.core.config.config_manager import ExperimentConfigManager
from sqa_system.experimentation.experiment_runner import ExperimentRunner
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class ExperimentRunCommandHandler(CommandHandler):
    """
    A command handler implementation that allows to run an experiment.
    """

    def __init__(self):
        self.evaluation_config_manager = ExperimentConfigManager()
        self.evaluation_config_manager.load_default_configs()

    def handle_command(self):
        weave.init("ma_mschneider_experiment_run")
        evaluation_configs = self.evaluation_config_manager.get_all_configs()
        if len(evaluation_configs) == 0:
            logger.info("No experiment configurations available")
            return

        choices = [
            Choice(config.uid, config.name) for config in evaluation_configs
        ]
        evaluation_uid = CLICommandHelper.get_selection(
            "Select which experiment to run",
            choices
        )
        if CLICommandHelper.is_exit_selection(evaluation_uid):
            return

        self.run_experiment(evaluation_uid)

    @weave.op()
    def run_experiment(self, evaluation_uid: str):
        """
        This method runs an experiment with the given evaluation_uid.

        Args:
            evaluation_uid (str): The evaluation_uid of the experiment to run.
        """
        try:
            evaluation_config = self.evaluation_config_manager.get_config(evaluation_uid)
            experiment_runner = ExperimentRunner(evaluation_config)
            experiment_runner.run()
        except APIKeyMissingError as e:
            CLICommandHelper.print(
                "API key is missing. Please set up the API Key using the Secret Manager: %s",
                str(e))
            return
        except Exception as e:
            logger.exception("The experiment exited with an error: %s", e)
            return

        CLICommandHelper.print("The experiment has fininished..")
        CLICommandHelper.print(
            f"The results can be found at {experiment_runner.results_folder_path}")


if __name__ == "__main__":
    handler = ExperimentRunCommandHandler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        help="The config id of the experiment to run")
    args = parser.parse_args()

    if args.config:
        config = handler.evaluation_config_manager.get_config_by_name(
            args.config)
        handler.run_experiment(config.uid)
