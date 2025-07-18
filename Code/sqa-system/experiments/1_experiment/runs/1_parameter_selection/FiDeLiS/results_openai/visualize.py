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
    "f02d2dd08d26874285efb4984f83f74c": "Base Config",
    "2d9c949bc961efb0aed4e666a7430bc3": "GPT-4o",
    "70a26d0835505a6606d675451bde6284": "top-k: 20",
    "73dde53993bc5c3476908adb39712c81": "question-augmentation",
    "98f15a84cf7e3c2f9a2b217baba81300": "reranking",
    "0573122b14aab6e53cd28a1650789d7a": "O3-mini",
    "5cc35c76e1450482fb90d9895d172063": "granite-embedding",
    "88fabdf30e3a0b5c1ddb2b12a1423691": "text-embedding-3-large",
    "c9ff59314b3c7e43d604e8f5883a340f": "llama3.1",
    "7c19d3906481f5966a9b850b7fb68593": "qwen2.5:14b",
    "0249e4bd52280915bf4c9c1574e46add": "top_n: 30",
    "405564a8e8bf353c3c61c9830011111d": "premature_stopping",
    "54d7ef07e354c35511e6fbbb7268415f": "top_k: 30",
    "56b2b0249b3ac9d1b1020bc46e0fcb08": "top_n: 20",
    "2186fdb6f13dfa2b7b9c97359f8e58e9": "alpha: 0.6",
    "9c9ee6e038989654f1a332467cbb0bba": "alpha: 0.1"
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
