This folder contains a collection of scripts:

- `paper_scrapping/`: This folder contains the scripts that were used to download the full text of the papers as PDF files. Then another script converts the PDF files to text files. 
- `run_basic_evaluation.py`: A script that requires a list of evaluator configurations and needs to be put into a folder containing prediction output files. It then runs the evaluators for the predictions and calculates the metric scores.
- `visualize.py`: A script to run the visualizer class of the SQA system to create plots and tables for evaluation results.