"""
HCA-CLIP Dataset Module
Only includes the three plant disease datasets
"""

from .plant_disease import PlantDisease
from .ai_challenger import AiChallenger
from .spd import SPD


dataset_list = {
    "plant_disease": PlantDisease,
    "ai_challenger": AiChallenger,
    "spd": SPD,
}


def build_dataset(dataset, root_path, shots):
    """Build dataset instance"""
    if dataset not in dataset_list:
        raise ValueError(f"Dataset '{dataset}' not found. Available datasets: {list(dataset_list.keys())}")
    return dataset_list[dataset](root_path, shots)
