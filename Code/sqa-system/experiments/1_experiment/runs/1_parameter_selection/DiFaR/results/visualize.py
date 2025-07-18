# A simple script to generate visualizations for the results of the
# Experiments.

# ----------------
# -- HOW TO USE --
# ----------------
# 1. Place this script in the same directory as the results of the
#    experiment(s) you want to visualize. It will automatically retrieve
#    all prediction files from the subfolders.
# 2. Run the script.
# (3.) Optionally you can change the variables inside of the script to change
#    what is plotted.

import os
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.experimentation.utils.visualizer.experiment_visualizer import (
    ExperimentVisualizer, ExperimentVisualizerSettings, PlotType)

# Here you replace the name of the configs with a unique
# name. Place the id on the left, and the name on the right.
CONFIG_TO_NAME_MAPPING: dict = {
    "79c3717badfbe834fb97ce5f07eb461d": "Base Config",
    "31ee6493808ef06a9fbb47aa619c4ec8": "distance-metric: IP",
    "6966b60ba281de9f7dc0a19081ba65de": "question_augmentation",
    "3afb87c518f0f42f89ff0dc6d09f6945": "distance-metric: L2",
    "61fa17b2025235f039cfbbd44dbea4bd": "reranking",
    "f7c75d1fa328fc2042a7f3d60ae80934": "text-embedding-3-large",
    "6bca5b9a09260b1efd118352e66d2a54": "n-results: 150",
    "c412eb9112d6ebdd8521a14c73576fd1": "n-results: 60",
    "d332fe8410c73e902b56f719fea19fd8": "n-results: 120",
    "dc6be09993f6c79a01af696bbac8d022": "granite-embedding",
    "381e0d2108aa4800c04a2c20f7ebbe17": "n-results: 90"
}
# The baseline config (or name if you replace it above)
# is highlighted in red in some plots. Should be a string.
BASELINE_CONFIG: str = "Base Config"


FPM = FilePathManager()
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
VISUALIZATIONS_DIR = FPM.combine_paths(
    CURRENT_DIRECTORY, "result_visualizations")
QA_DATASET_PATH = FPM.combine_paths(
    FPM.get_parent_directory(CURRENT_DIRECTORY, 5),
    "qa_datasets",
    "qa_datasets",
    "reduced",
    "reduced_deep_distributed_graph_dataset.csv"
)


def plot_average_metrics_per_config():
    """
    Generates a plot with the y-axis being the average of the metrics,
    the x-axis being the names of the metrics and the hue being the
    configuration hashes.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            baseline_config=BASELINE_CONFIG,
            save_folder_path=VISUALIZATIONS_DIR,
            qa_file_path=QA_DATASET_PATH,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG,
            PlotType.TABLE
        ]
    )


def plot_average_metrics_by_column():
    """
    This function generates a plot with the y-axis beeing the average of the 
    metrics, the x-axis being the names of the metrics and the hue being the
    unique values of the column to group by.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            save_folder_path=VISUALIZATIONS_DIR,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            qa_file_path=QA_DATASET_PATH,
            # It is recommended to add the config to visualize here else it will
            # generate many plots for each config that it finds.
            configs_to_visualize=[]
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_BY_COLUMN,
        ],
        column_to_group_by="retrieval_operation"
    )


def plot_metric_by_column():
    """
    This function generates per metric a plot with the metrics average value
    being on the y-axis, the unique values of the column on the x-axis and
    the hue being the configuration hashes.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            file_type="png",
            save_folder_path=VISUALIZATIONS_DIR,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            qa_file_path=QA_DATASET_PATH,
            # Here you need to define the names of the metrics you want to
            # visualize. It has to be inside of a dictionary.
            metrics={
                "retrieval_operation": [
                    "recall_triples",
                ]
            }
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_PER_CONFIG
        ],
        column_to_group_by="retrieval_operation"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_PER_CONFIG
        ],
        column_to_group_by="use_case"
    )


if __name__ == '__main__':
    plot_average_metrics_per_config()
    plot_metric_by_column()
