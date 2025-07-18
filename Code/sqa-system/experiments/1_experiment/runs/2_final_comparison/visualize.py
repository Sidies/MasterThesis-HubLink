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
    "f7bf83ab9c8915163ea44bfe1d533dbc": "HubLink (Fast)",
    "80a54da81afa95067d08d3c2fa4e5bb1": "DiFaR",
    "8c4e0c92bde8c2ba35c82cfa1d244646": "HubLink (Traverse)",
    "02b073d17825013f38af6048fc7db0ef": "Mindmap",
    "cea3fa9c232ebfce12b34889ebf7e3b8": "FiDeLiS",
    "dd1cd4d101ca38fc45fd61d21d81d8d4": "HubLink (Open)",
    "f5d2a4751b650feada14d589c657be6b": "HubLink (Direct)"
}
# The base config (or name if you replace it above)
# is highlighted in red in some plots. Should be a string.
BASELINE_CONFIG: str = "Baseline"


FPM = FilePathManager()
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
VISUALIZATIONS_DIR = FPM.combine_paths(
    CURRENT_DIRECTORY, "result_visualizations")
# The path to the QA dataset which is used to fetch the classifications
# of each question like the retrieval operation or answer type.
# You need to adapt this location if the dataset is located elsewhere.
QA_DATASET_PATH = FPM.combine_paths(
    FPM.get_parent_directory(CURRENT_DIRECTORY, 3),
    "qa_datasets",
    "qa_datasets",
    "full",
    "deep_distributed_graph_dataset.csv"
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
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            save_folder_path=VISUALIZATIONS_DIR,
            qa_file_path=QA_DATASET_PATH,
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
    This function generates a plot with the y-axis being the average of the 
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
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_BY_COLUMN,
        ],
        column_to_group_by="use_case"
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
            file_type="pdf",
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


def create_tables_by_column():
    """
    This function generates per configuration a table with the
    unique values of the column to group by and the average values
    of the metrics.
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
            metrics=None
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="retrieval_operation"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="use_case"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="semi-typed"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="hops"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="graph_representation"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="answer_type"
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.METRIC_BY_COLUMN_TABLE
        ],
        column_to_group_by="condition_type"
    )

def plot_sustainability_table():
    """
    This function generates a table calculating the sustainability
    metrics.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            save_folder_path=VISUALIZATIONS_DIR,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            metrics=None
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.SUSTAINABILITY_TABLE
        ],
        column_to_group_by="recall_triples"
    )
    
def plot_metrics_in_vector_format():
    """
    Generates plots in vector format.
    """
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            save_folder_path=VISUALIZATIONS_DIR,
            qa_file_path=QA_DATASET_PATH,
            file_type="svg",
            metrics={
                "Retrieval Operation": [
                    "hit@10_triples",
                    "map@10_triples",
                    "mrr@10_triples",
                    "exact_match@10_triples",
                    "recall_triples",
                    "precision_triples",
                    "f1_triples",
                ]
            },
            new_metric_names_mapping={
                "hit@10_triples": "Hit@10",
                "map@10_triples": "MAP@10",
                "mrr@10_triples": "MRR@10",
                "exact_match@10_triples": "EM@10",
                "recall_triples": "Recall",
                "precision_triples": "Precision",
                "f1_triples": "F1"
            }
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG
        ]
    )
    
    visualizer = ExperimentVisualizer(
        ExperimentVisualizerSettings(
            data_folder_path=CURRENT_DIRECTORY,
            should_print=False,
            should_save_to_file=True,
            baseline_config=BASELINE_CONFIG,
            config_to_name_mapping=CONFIG_TO_NAME_MAPPING,
            save_folder_path=VISUALIZATIONS_DIR,
            qa_file_path=QA_DATASET_PATH,
            file_type="svg",
            metrics={
                "Answer Generation": [
                    "factual_correctness(mode=recall)",
                    "factual_correctness(mode=precision)",
                    "rouge_1_recall",
                    "rouge_1_precision",
                    "semantic_similarity",
                    "non_llm_string_similarity"
                ]
            },
            new_metric_names_mapping={
                "factual_correctness(mode=recall)": "FC (Recall)",
                "factual_correctness(mode=precision)": "FC (Precision)",
                "rouge_1_recall": "ROUGE-1-Recall",
                "rouge_1_precision": "ROUGE-1-Precision",
                "semantic_similarity": "Semantic Similarity",
                "non_llm_string_similarity": "String Similarity"
            },
            configs_to_visualize=[
                "HubLink (Traverse)",
                "DiFaR",
                "Mindmap",
                "FiDeLiS"
            ]
        )
    )
    visualizer.run(
        plots_to_generate=[
            PlotType.AVERAGE_METRICS_PER_CONFIG
        ]
    )


if __name__ == '__main__':
    plot_average_metrics_per_config()
    plot_metric_by_column()
    create_tables_by_column()
    plot_sustainability_table()
    plot_average_metrics_by_column()
    plot_metrics_in_vector_format()