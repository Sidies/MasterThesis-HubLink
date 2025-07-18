import os

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.experimentation.experiment_config_builder import ExperimentConfigBuilder
from sqa_system.experimentation.experiment_runner import ExperimentRunner, ExperimentRunnerSettings


def main():
    """Main function to run the experiment."""
    # -----------------------
    # -- Prepare the paths --
    # -----------------------
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    results_path = fpm.combine_paths(current_directory, "results")
    qa_dataset_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 4),
        "qa_datasets",
        "qa_datasets",
        "full",
        "deep_distributed_graph_dataset.csv"
    )
    evaluator_configs_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 4),
        "evaluator_configs.json"
    )
    configuration = fpm.combine_paths(
        current_directory,
        "fidelis_config.json",
    )

    # --------------------------------
    # -- Build the ExperimentConfig --
    # --------------------------------
    exp_config_builder = ExperimentConfigBuilder()
    exp_config_builder.set_baseline_by_path(configuration)
    exp_config_builder.load_evaluators_from_path(evaluator_configs_path)
    experiment_config = exp_config_builder.build()

    # ------------------------
    # -- Run the Experiment --
    # ------------------------
    runner = ExperimentRunner(
        experiment_config=experiment_config,
        settings=ExperimentRunnerSettings(
            results_folder_path=fpm.combine_paths(
                results_path, experiment_config.config_hash),
            qa_data_path=qa_dataset_path,
            debugging=True,
            log_to_results_folder=True,
            weave_project_name="kastel-sdq-meta-research/ma_mschneider_1_exp_comparison",
            number_of_workers=1,
            number_of_processes=1,
        ),
    )

    results = runner.run()
    print(results.head())


if __name__ == '__main__':
    main()
