
import os
from sqa_system.experimentation.file_evaluator.file_evaluator import (
    FileEvaluator, FilePathManager)

# ----- HOW TO USE THIS SCRIPT -----
# This script is used to evaluate all prediction files in a folder and the subfolders.
# This is helpful to correct metrics or add more metrics after the experiment is done.
# 
# 1. Add this script to the folder where you have your prediction files.
# 2. Create a Evaluator Config file and add it to the same folder.
# 3. Run this script.
# ----- END OF HOW TO USE THIS SCRIPT -----

def main():
    """Runs the LLM evaluation for all prediction files in the folder."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path_manager = FilePathManager()
    file_evaluator = FileEvaluator()
    evaluator_config_path = file_path_manager.combine_paths(
        file_path_manager.get_parent_directory(current_directory, 0),
        "evaluator_configs.json"
    )
    file_evaluator.update_prediction_files_in_folder(
        folder_path=current_directory,
        evaluator_config_path=evaluator_config_path
    )
    
    
if __name__ == "__main__":
    main()