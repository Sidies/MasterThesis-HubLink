import os
from typing import Dict, List, Any, Tuple, Optional
import json

# --- HOW TO USE THIS SCRIPT ---
# 1. Place in the parent directory of another directory named 'taxonomy_applications' which contains JSON files
#    with the taxonomy classifications. Make sure that inside of this directory, there is a file named
#    'main_taxonomy_application.json' which contains the classifications from your own taxonomy.
# 2. Ensure your JSON files are formatted correctly, with each file containing a list of dictionaries.
# 3. Each dictionary should have a "research question" and "annotations" field.
# 4. Run the script. It will load the JSON files, calculate the classification delta, and save the result to a text file.
# 5. The output will be saved in the same directory as the script, named "classification_delta.txt".
# --- END OF HOW TO USE THIS SCRIPT ---


def calculate_equivalence_classes(classification_results: Dict[Any, Any]) -> int:
    """
    Calculates the number of equivalence classes for the taxonomy classifications.
    Here, an equivalence class is a unique classification to one or more objects.

    Args:
        classification_results (Dict[Any, Any]): A dictionary where the keys are
            object identifiers and values are the classification assigned by the taxonomy.

    Returns:
        int: The number of equivalence classes.
    """
    if not classification_results:
        return 0

    unique_classifications = set(classification_results.values())
    return len(unique_classifications)


def calculate_classification_delta(results_evaluated_tax: Dict[Any, Any],
                                   results_previous_taxonomies: List[Dict[Any, Any]]) -> str:
    """
    This calculates the classification delta to measure the significance of the
    taxonomy. It is implemented based on the formula:
    classification_delta(C, T, R) = (|~C| - max(T in T) |~T|) / |~C|

    """
    if not results_evaluated_tax:
        print("Error: Evaluated taxonomy results are empty or could not be loaded.")
        return float('nan')

    if not results_previous_taxonomies:
        print("Error: No valid previous taxonomies provided or loaded for comparison.")
        return float('nan')

    num_equiv_classes_evaluated = calculate_equivalence_classes(
        results_evaluated_tax)

    # To avoid division by zero
    if num_equiv_classes_evaluated == 0:
        print("Error: Evaluated taxonomy resulted in 0 equivalence classes.")
        return float('nan')

    num_equiv_classes_previous = []
    valid_previous_count = 0
    for _, results in enumerate(results_previous_taxonomies):
        if not results:
            num_equiv_classes_previous.append(0)
        else:
            equiv_classes = calculate_equivalence_classes(results)
            num_equiv_classes_previous.append(equiv_classes)
            valid_previous_count += 1

    if valid_previous_count == 0:
        print("Error: None of the previous taxonomies could be loaded successfully.")
        return float('nan')

    max_equiv_classes_previous = max(
        num_equiv_classes_previous) if num_equiv_classes_previous else 0

    denominator = num_equiv_classes_evaluated - max_equiv_classes_previous
    nominator = num_equiv_classes_evaluated
    classification_delta = (num_equiv_classes_evaluated -
                            max_equiv_classes_previous) / num_equiv_classes_evaluated

    return f"classification delta: {classification_delta:.3f} ({denominator}/{nominator})"


def load_classifications_from_json(file_path: str) -> Optional[Dict[str, Tuple]]:
    """
    Loads the classifications from a taxonomy from a JSON file.
    This function expects a JSON file which contains a list of dictionaries, 
    where each contains a "research question", and "annotations" field.

    Args:
        file_path (str): The path to the JSON file containing the classifications.

    Returns:
        Optional[Dict[str, Tuple]]: A dictionary where the keys are research questions and the values
            are tuples of sorted annotations. Returns None if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            classifications = {}
            if not isinstance(data, list):
                print(
                    f"Error, the JSON file at {file_path} does not contain a list.")
                return None

            for item in data:
                if not isinstance(item, dict):
                    print(
                        f"Skipping, the item in the JSON file at {file_path} is not a dictionary.")
                    continue
                if "doi" not in item or "annotations" not in item:
                    print(
                        f"Skipping, the item in the JSON file at {file_path} does not contain 'doi' or 'annotations'.")
                    continue
                if not isinstance(item["annotations"], list):
                    print(
                        f"Skipping, the 'annotations' field in the JSON file at {file_path} is not a list.")
                    continue

                research_question = item.get(
                    'research_question', '').strip().lower()
                annotations = tuple(sorted(item["annotations"]))
                classifications[research_question] = annotations

            return classifications

    except Exception as e:
        print(f"Error loading classifications from {file_path}: {e}")
        return None


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_taxonomy_file = os.path.join(
        current_dir, "taxonomy_applications/main_taxonomy_application.json")
    print(f"Loading main taxonomy classifications from: {main_taxonomy_file}")
    evaluated_results = load_classifications_from_json(main_taxonomy_file)
    if evaluated_results is None:
        print("Error: Failed to load classifications.")
        exit(1)

    all_previous_results = []
    # load all json files from the current directory but not the main taxonomy
    for file in os.listdir(os.path.join(current_dir, "taxonomy_applications")):
        if file.endswith(".json") and file != "main_taxonomy_application.json":
            file_path = os.path.join(
                current_dir, "taxonomy_applications", file)
            print(
                f"Loading previous taxonomy classifications from: {file_path}")
            previous_results = load_classifications_from_json(file_path)
            if previous_results is not None:
                all_previous_results.append(previous_results)

    valid_previous_results = [
        res for res in all_previous_results if res is not None]
    delta = calculate_classification_delta(
        evaluated_results, valid_previous_results)

    print(f"\nCalculated Classification Delta: {delta}")

    output_file = os.path.join(
        current_dir, "classification_delta.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(delta)
