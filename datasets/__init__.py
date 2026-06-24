from .custom_dataset import PairedImageDataset
from .rl_dataset_adapter import RLDatasetAdapter

# Alias for backward/forward compatibility if referenced elsewhere
MultiModalDataset = PairedImageDataset

__all__ = ["PairedImageDataset", "MultiModalDataset", "RLDatasetAdapter"]
