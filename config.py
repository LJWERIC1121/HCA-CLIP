"""
Configuration file for HCA-CLIP
Update these paths according to your environment
"""

import os

# ========== Data Paths ==========
# Root directory for all datasets
DATA_ROOT = "D:/dataset/FSIC"

# Individual dataset paths (relative to DATA_ROOT)
DATASET_PATHS = {
    "plant_disease": "Plant_disease",
    "ai_challenger": "Ai_Challenger_2018",
    "spd": "SPD",
}

# ========== Model Paths ==========
# Directory containing pre-trained CLIP models
MODEL_CACHE_DIR = "./model/clip"

# ========== Output Paths ==========
# Directory for training logs and results
LOG_ROOT = "./result/log"

# Directory for saved model checkpoints
CHECKPOINT_DIR = "./result/checkpoints"

# ========== Training Settings ==========
# Default number of workers for data loading
NUM_WORKERS = 8

# Whether to use GPU
USE_CUDA = True

# ========== Reproducibility ==========
# Default random seed
DEFAULT_SEED = 2024
