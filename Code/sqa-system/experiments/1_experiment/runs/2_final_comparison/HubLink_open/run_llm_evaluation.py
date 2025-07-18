
import os
from sqa_system.experimentation.file_evaluator.file_evaluator import (
    FileEvaluator, FilePathManager)


def main():
    """Runs the LLM evaluation for all prediction files in the folder."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    os.environ["WEAVE_DISABLED"] = "true"
    file_path_manager = FilePathManager()
    file_evaluator = FileEvaluator()
    evaluator_config_path = file_path_manager.combine_paths(
        file_path_manager.get_parent_directory(current_directory, 2),
        "evaluator_configs.json"
    )
    file_evaluator.update_prediction_files_in_folder(
        folder_path=current_directory,
        evaluator_config_path=evaluator_config_path
    )


if __name__ == "__main__":
    main()
