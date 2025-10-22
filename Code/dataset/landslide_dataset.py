import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from typing import List, Tuple, Dict, Optional
import random
from PIL import Image
from tqdm import tqdm
import shutil
from datetime import datetime
import json


class LandslideDataset(Dataset):
    """
    Landslide change detection dataset (optimized)
    Supports efficient loading, reduces memory usage, accelerates training,
    and supports stratified sampling by location.
    Handles naming format: "location_row_col.tif" or "location_row_col_augmethod.tif"
    """
    
    def __init__(self, 
                 root_dir: str, 
                 mode: str = 'train', 
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 42,
                 exclude_features: Optional[List[str]] = None,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 use_png_labels: bool = False,
                 use_data_cache: bool = False,
                 cache_dir: Optional[str] = None,
                 preprocess_npz: bool = False):
        """Initialize dataset"""
        self.root_dir = root_dir
        self.mode = mode
        self.split_ratios = split_ratios
        self.seed = seed
        self.exclude_features = exclude_features if exclude_features else []
        self.transform = transform
        self.target_transform = target_transform
        self.use_png_labels = use_png_labels
        
        self.use_data_cache = use_data_cache
        self.cache_dir = cache_dir or os.path.join(root_dir, 'npz_cache')
        self.preprocess_npz = preprocess_npz
        self._init_cache_dir()
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.sample_ids = self._get_sample_ids()
        self.feature_types = self._get_feature_types()
        
        self.location_groups = self._group_samples_by_location()
        print(f"Detected location groups: {', '.join(self.location_groups.keys())}")
        
        self.sample_names = self.sample_ids.copy()
        self.sample_info = self._precompute_sample_info()
        
        if mode != 'full':
            self._split_dataset()

        self.feature_cache = {}
        self.label_cache = {}
        self.cache_max_size = 50
        
        if self.preprocess_npz:
            self._check_and_preprocess_npz()
        
        print(f"Initialized {mode} dataset:")
        print(f"  Number of samples: {len(self.sample_ids)}")
        print(f"  Features used: {', '.join(self.feature_types)}")
        print(f"  Excluded features: {', '.join(self.exclude_features) if self.exclude_features else 'None'}")
        print(f"  Data format: {'NPZ' if preprocess_npz else 'TIFF'}")
        print(f"  Cache mode: {'Limited cache' if use_data_cache else 'No cache'}")
        print(f"  Number of locations: {len(self.location_groups)}")
    
    def _init_cache_dir(self):
        """Initialize NPZ cache directory"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Created NPZ cache directory: {self.cache_dir}")
    
    def _get_sample_ids(self) -> List[str]:
        """Get all sample IDs"""
        label_dir = os.path.join(self.root_dir, 'label')
        if not os.path.exists(label_dir):
            raise ValueError(f"Label directory does not exist: {label_dir}")
        
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
        sample_ids = [os.path.splitext(f)[0] for f in label_files]
        return list(set(sample_ids))
    
    def _group_samples_by_location(self) -> Dict[str, List[str]]:
        """Group samples by location"""
        groups = {}
        for sample_id in self.sample_ids:
            parts = sample_id.split('_')
            if len(parts) >= 3:
                location = parts[0]
                if location not in groups:
                    groups[location] = []
                groups[location].append(sample_id)
            else:
                print(f"Warning: Sample ID format unexpected: {sample_id}, assigned to 'other'")
                if 'other' not in groups:
                    groups['other'] = []
                groups['other'].append(sample_id)
        return groups
    
    def _get_feature_types(self) -> List[str]:
        """Get available feature types"""
        data_dir = os.path.join(self.root_dir, 'data')
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        all_types = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        return [t for t in all_types if t not in self.exclude_features]
    
    def _precompute_sample_info(self) -> Dict[str, Dict]:
        """Precompute feature paths and band info for each sample"""
        sample_info = {}
        for sample_id in self.sample_ids:
            info = {
                'feature_paths': {},
                'band_counts': {},
                'total_bands': 0,
                'shape': None
            }
            for feature_type in self.feature_types:
                tif_path = os.path.join(self.root_dir, 'data', feature_type, f"{sample_id}.tif")
                if not os.path.exists(tif_path):
                    raise FileNotFoundError(f"Sample file does not exist: {tif_path}")
                with rasterio.open(tif_path) as src:
                    info['feature_paths'][feature_type] = tif_path
                    info['band_counts'][feature_type] = src.count
                    info['total_bands'] += src.count
                    info['shape'] = (src.height, src.width)
            sample_info[sample_id] = info
        return sample_info
    
    def _split_dataset(self):
        """Split dataset using stratified sampling by location"""
        train_ids, val_ids, test_ids = [], [], []
        for location, samples in self.location_groups.items():
            random.shuffle(samples)
            total = len(samples)
            train_size = int(self.split_ratios[0] * total)
            val_size = int(self.split_ratios[1] * total)
            train = samples[:train_size]
            val = samples[train_size:train_size+val_size]
            test = samples[train_size+val_size:]
            train_ids.extend(train)
            val_ids.extend(val)
            test_ids.extend(test)
            print(f"  {location} sample allocation: train={len(train)}, val={len(val)}, test={len(test)}")
        if self.mode == 'train':
            self.sample_ids = train_ids
            self.sample_names = train_ids.copy()
        elif self.mode == 'val':
            self.sample_ids = val_ids
            self.sample_names = val_ids.copy()
        elif self.mode == 'test':
            self.sample_ids = test_ids
            self.sample_names = test_ids.copy()
    
    def _save_feature_types(self):
        """Save current feature types to cache"""
        feature_info_path = os.path.join(self.cache_dir, 'feature_types.json')
        feature_set = set(self.feature_types)
        with open(feature_info_path, 'w') as f:
            json.dump(list(feature_set), f)
        print(f"Saved feature types to: {feature_info_path}")
    
    def _load_feature_types(self) -> Optional[set]:
        """Load saved feature types from cache"""
        feature_info_path = os.path.join(self.cache_dir, 'feature_types.json')
        if not os.path.exists(feature_info_path):
            return None
        with open(feature_info_path, 'r') as f:
            feature_list = json.load(f)
        return set(feature_list)
    
    def _check_feature_consistency(self) -> bool:
        """Check if current features match cached features"""
        cached_features = self._load_feature_types()
        current_features = set(self.feature_types)
        if cached_features is None:
            return False
        return cached_features == current_features
    
    def _clean_npz_cache(self):
        """Clear existing NPZ cache and feature info"""
        if not os.path.exists(self.cache_dir):
            return
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.npz'):
                os.remove(os.path.join(self.cache_dir, filename))
        feature_info_path = os.path.join(self.cache_dir, 'feature_types.json')
        if os.path.exists(feature_info_path):
            os.remove(feature_info_path)
        print(f"Cleared NPZ cache directory: {self.cache_dir}")
    
    def _check_and_preprocess_npz(self):
        """Check feature consistency and preprocess if needed"""
        has_npz_files = any(f.endswith('.npz') for f in os.listdir(self.cache_dir)) if os.path.exists(self.cache_dir) else False
        features_consistent = self._check_feature_consistency()
        if has_npz_files and not features_consistent:
            print("Detected feature type change, reprocessing NPZ files...")
            self._clean_npz_cache()
        self._preprocess_to_npz()
        self._save_feature_types()
    
    def _preprocess_to_npz(self):
        """Convert TIFF to NPZ for faster reading"""
        print(f"Converting TIFF to NPZ (cached in {self.cache_dir})...")
        for sample_id in tqdm(self.sample_ids, desc="Preprocessing NPZ"):
            npz_path = os.path.join(self.cache_dir, f"{sample_id}.npz")
            if os.path.exists(npz_path):
                continue
            info = self.sample_info[sample_id]
            combined_features = np.empty((info['total_bands'], info['shape'][0], info['shape'][1]), dtype=np.float32)
            current_band = 0
            for feature_type in self.feature_types:
                with rasterio.open(info['feature_paths'][feature_type]) as src:
                    data = src.read()
                    bands = info['band_counts'][feature_type]
                    combined_features[current_band:current_band+bands] = data
                    current_band += bands
            if self.use_png_labels:
                label_path = os.path.join(self.root_dir, 'label_png', f"{sample_id}.png")
                label = np.array(Image.open(label_path))
                label = (label > 0).astype(np.uint8)
            else:
                label_path = os.path.join(self.root_dir, 'label', f"{sample_id}.tif")
                with rasterio.open(label_path) as src:
                    label = src.read(1)
            np.savez_compressed(npz_path, features=combined_features, label=label)
    
    def _load_from_npz(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from NPZ cache"""
        npz_path = os.path.join(self.cache_dir, f"{sample_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file does not exist: {npz_path} (set preprocess_npz=True first)")
        data = np.load(npz_path)
        return data['features'], data['label']
    
    def _load_from_tif(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from TIFF"""
        info = self.sample_info[sample_id]
        combined_features = np.empty((info['total_bands'], info['shape'][0], info['shape'][1]), dtype=np.float32)
        current_band = 0
        for feature_type in self.feature_types:
            with rasterio.open(info['feature_paths'][feature_type]) as src:
                data = src.read()
                bands = info['band_counts'][feature_type]
                combined_features[current_band:current_band+bands] = data
                current_band += bands
        if self.use_png_labels:
            label_path = os.path.join(self.root_dir, 'label_png', f"{sample_id}.png")
            label = np.array(Image.open(label_path))
            label = (label > 0).astype(np.uint8)
        else:
            label_path = os.path.join(self.root_dir, 'label', f"{sample_id}.tif")
            with rasterio.open(label_path) as src:
                label = src.read(1)
        return combined_features, label
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently load a single sample"""
        sample_id = self.sample_ids[idx]
        if self.use_data_cache and sample_id in self.feature_cache:
            features = self.feature_cache[sample_id]
            label = self.label_cache[sample_id]
        else:
            if self.preprocess_npz:
                features_np, label_np = self._load_from_npz(sample_id)
            else:
                features_np, label_np = self._load_from_tif(sample_id)
            features = torch.from_numpy(features_np).float()
            label = torch.from_numpy(label_np).long()
            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                label = self.target_transform(label)
            if self.use_data_cache:
                if len(self.feature_cache) >= self.cache_max_size:
                    oldest_key = next(iter(self.feature_cache.keys()))
                    del self.feature_cache[oldest_key]
                    del self.label_cache[oldest_key]
                self.feature_cache[sample_id] = features
                self.label_cache[sample_id] = label
        return features, label
    
    def get_feature_info(self) -> Dict[str, int]:
        """Get feature info"""
        feature_info = {}
        if not self.sample_ids:
            return feature_info
        first_sample = self.sample_ids[0]
        for feature_type in self.feature_types:
            feature_info[feature_type] = self.sample_info[first_sample]['band_counts'][feature_type]
        return feature_info
    
    def clear_cache(self):
        """Manually clear cache"""
        self.feature_cache.clear()
        self.label_cache.clear()
        print("Dataset cache cleared")
    
    def get_location_distribution(self) -> Dict[str, int]:
        """Get sample count per location"""
        distribution = {}
        for sample_id in self.sample_ids:
            parts = sample_id.split('_')
            location = parts[0] if len(parts) >= 3 else 'other'
            distribution[location] = distribution.get(location, 0) + 1
        return distribution


# Example usage
if __name__ == "__main__": 
    dataset_root = r"D:\lb\myCode\Landslide_detection\Data\cut_max_1.0"
    exclude_features = ['1m', 'dem']
    seed = 42
    
    full_dataset = LandslideDataset(
        root_dir=dataset_root,
        mode='full',
        exclude_features=exclude_features,
        seed=seed,
        use_data_cache=True,
        preprocess_npz=True,
        cache_dir=os.path.join(dataset_root, 'npz_cache')
    )
    
    feature_info = full_dataset.get_feature_info()
    print("\nFeature info:")
    for feature, bands in feature_info.items():
        print(f"  {feature}: {bands} bands")
    
    train_dataset = LandslideDataset(
        root_dir=dataset_root,
        mode='train',
        exclude_features=exclude_features,
        seed=seed,
        use_data_cache=True,
        preprocess_npz=True,
        cache_dir=os.path.join(dataset_root, 'npz_cache')
    )
    
    val_dataset = LandslideDataset(
        root_dir=dataset_root,
        mode='val',
        exclude_features=exclude_features,
        seed=seed,
        use_data_cache=True,
        preprocess_npz=True,
        cache_dir=os.path.join(dataset_root, 'npz_cache')
    )
    
    test_dataset = LandslideDataset(
        root_dir=dataset_root,
        mode='test',
        exclude_features=exclude_features,
        seed=seed,
        use_data_cache=True,
        preprocess_npz=True,
        cache_dir=os.path.join(dataset_root, 'npz_cache')
    )
    
    print("\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    print("\nTrain location distribution:")
    train_dist = train_dataset.get_location_distribution()
    for loc, count in train_dist.items():
        print(f"  {loc}: {count} samples")
        
    print("\nValidation location distribution:")
    val_dist = val_dataset.get_location_distribution()
    for loc, count in val_dist.items():
        print(f"  {loc}: {count} samples")
    
    start_time = datetime.now()
    for i in range(10):
        features, label = train_dataset[i]
    end_time = datetime.now()
    print(f"\nLoading 10 samples took: {(end_time - start_time).total_seconds():.2f}s")
    print(f"  Features shape: {features.shape}")
    print(f"  Label shape: {label.shape}")
