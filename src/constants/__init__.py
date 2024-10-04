from pathlib import Path
import glob

# Define the base directory path relative to the local file
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define paths to the segmentation and classification CXR image dataset 
SEG_DATA_DIR = BASE_DIR / 'data' / 'segmentation'
CLF_DATA_DIR = BASE_DIR / 'data' / 'tb_classification'
PROC_DATA_DIR = BASE_DIR / 'data'/ 'processed'
# paths to saved model files
SEG_MODEL_DIR = BASE_DIR / 'models'/ 'segmentation'
SEG_MODEL_PATH = next(SEG_MODEL_DIR.glob('*.pth'))
CLF_MODEL_DIR = BASE_DIR / 'models' / 'classification'

#config paths
CONFIG_DIR = BASE_DIR / 'configs'
IMG_RESIZE_CFG = CONFIG_DIR / 'model_image_size.yaml'

# experiment logging 
EXPERIMENT_LOG_FILE = BASE_DIR / 'experiment_logs'
