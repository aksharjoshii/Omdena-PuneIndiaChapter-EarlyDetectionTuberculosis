import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import json
import random
import string
from src.constants import IMG_RESIZE_CFG, EXPERIMENT_LOG_FILE, CLF_MODEL_DIR

def calculate_class_weights(df):
    """Calculates normalized class weights for a binary classification problem.

    Args:
        df: The Pandas DataFrame containing the data.
        label_column: The name of the column containing the labels.

    Returns:
        A dictionary containing the class weights.
    """

    # Count the occurrences of each class
    class_counts = df['label'].value_counts()

    # Calculate the total number of samples
    total_samples = len(df)

    # Calculate the class weights
    class_weights = {
        class_label: total_samples / (2 * class_count)
        for class_label, class_count in class_counts.items()
    }

    # Normalize the class weights
    total_weights = sum(class_weights.values())
    normalized_class_weights = {
        class_label: weight / total_weights
        for class_label, weight in class_weights.items()
    }

    return normalized_class_weights



def update_config(config, yaml_file=IMG_RESIZE_CFG):
    with open(yaml_file, 'r') as file:
        size_mapping = yaml.safe_load(file)
    # Create a random experiment if not provided
    if config['experiment_name'] is None:
        config['experiment_name'] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    

    model_name = config.get('model_name')
    if model_name in size_mapping:
        config.update(size_mapping[model_name])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Check if 'imb_criterion' is not 'FocalLoss' and update 'focal_gamma' accordingly
    if not config.get('imb_criterion'):
        config['focal_gamma'] = 0.0

    return config
    

def read_yaml(file_path):
  """
  Reads a YAML file and returns the data as a dictionary.

  Args:
      file_path (str): The path to the YAML file.

  Returns:
      dict: The data from the YAML file.

  Raises:
      FileNotFoundError: If the YAML file is not found.
      YAMLError: If there's an error parsing the YAML file.
  """
  try:
    with open(file_path, 'r') as file:
      data = yaml.safe_load(file)
    return data
  except FileNotFoundError as e:
    raise FileNotFoundError(f"Error: YAML file not found at {file_path}") from e
  except yaml.YAMLError as e:
    raise ValueError(f"Error: Failed to parse YAML file {file_path}") from e




    
def convert_numpy_to_python(obj):
    """
    Recursively convert numpy types to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(i) for i in obj]
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    return obj

def save_experiment(experiment_name: str, history: dict, pred_data: dict, path_to_save=EXPERIMENT_LOG_FILE) -> None:
    """
    Saves the experiment results to a JSON file.

    Args:
        experiment_name (str): The name of the experiment.
        histories (dict): A dictionary containing the training and validation history.
        pred_data (dict): A dictionary containing the predicted data.
        path_to_save (str): The path to save the JSON file.
    """
    json_path = EXPERIMENT_LOG_FILE / 'experiments.json'
    experiment_data = {
        "history": convert_numpy_to_python(history),
        "pred_data": convert_numpy_to_python(pred_data)
    }

    try:
        with open(json_path, "r") as f:
            experiments = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        experiments = {}

    experiments[experiment_name] = experiment_data

    with open(path_to_save, "w") as f:
        json.dump(experiments, f, indent=3)



def save_model(config, best_metric_value, best_model_state_dict):
    trained_model_dir = CLF_MODEL_DIR
    state_dict_name = f"{config['model_name']}_{config['validation_metric']}_{best_metric_value:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_weights.pth"
    state_dict_path = os.path.join(trained_model_dir, state_dict_name)
    torch.save(best_model_state_dict, state_dict_path)
    return state_dict_name


CSV_PATH = EXPERIMENT_LOG_FILE / 'experiment_logs.csv'
def save_model_state(config, new_val_score, best_model_state_dict):
    # Check if the CSV file exists
    if not os.path.isfile(CSV_PATH):
        state_dict_name = save_model(config, new_val_score, best_model_state_dict)
        return state_dict_name  # If the file doesn't exist, treat it as a new model and return True

    # Load the CSV file into a DataFrame
    df = pd.read_csv(CSV_PATH)
    
    # Extract variable names from the config dictionary
    model_name = config['model_name']
    validation_metric = config['validation_metric']
   

   # Filter the DataFrame for the given parameters
    if validation_metric in ['precision', 'recall', 'f1-score', 'specificity']:
        filtered_df = df[
            (df['model_name'] == model_name) &
            (df['validation_metric'] == validation_metric) 
        ]
    else:
        filtered_df = df[
            (df['model_name'] == model_name) &
            (df['validation_metric'] == validation_metric)
        ]

    # If the filtered DataFrame is empty, save the model and return True
    if filtered_df.empty:
        state_dict_name = save_model(config, new_val_score, best_model_state_dict)
        return state_dict_name
    
    # Get the maximum best_val_score for the matching rows
    max_best_val_score = filtered_df['best_val_score'].max() if not filtered_df.empty else float('-inf')
    
    # Compare new_val_score with max_best_val_score
    if new_val_score > max_best_val_score:
        state_dict_name = save_model(config, new_val_score, best_model_state_dict)
        return state_dict_name
    
    return None