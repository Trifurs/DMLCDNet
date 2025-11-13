import sys
import os
import torch
import logging
import warnings
import numpy as np
import multiprocessing
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from utils import *  

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

TARGET_LAYERS = {
    'decoder.conv_block1',
    'decoder.conv_block2',
    'decoder.conv_block3',
    'fixed_encoder.conv2',
    'fixed_encoder.conv3',
    'fixed_encoder.shared_conv',
    'fixed_encoder.change_detector',
    'fusion.attention'
}

VISUALIZATION_METHODS = {
    'first_three': "Use first three bands",
    'max_variance': "Use three bands with maximum variance",
    'pca': "Reduce to three bands using PCA",
    'grayscale_mean': "Average of all bands (grayscale)",
    'grayscale_max': "Maximum of all bands (grayscale)"
}

DEFAULT_VISUALIZATION = 'pca'

class CAMVisualizer:
    """Class Activation Mapping (CAM) visualization tool for models"""
    
    def __init__(self, model, feature_names: List[str], device: torch.device, 
                 vis_method: str = DEFAULT_VISUALIZATION):
        """Initialize CAM visualizer"""
        self.model = self._get_unwrapped_model(model).eval()
        self.feature_names = feature_names
        self.device = device
        self.hooks = []
        self.feature_maps = {}
        self.vis_method = vis_method
        
        if self.vis_method not in VISUALIZATION_METHODS:
            raise ValueError(f"Invalid visualization method: {self.vis_method}, available methods: {list(VISUALIZATION_METHODS.keys())}")
        
        self._register_hooks()
    
    def _get_unwrapped_model(self, model):
        """Get original model wrapped by DataParallel or DistributedDataParallel"""
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            return model.module
        return model
    
    def _register_hooks(self):
        """Register hooks for key model layers, focusing on target layers"""
        if hasattr(self.model.fixed_encoder, 'shared_conv'):
            self.hooks.append(self.model.fixed_encoder.shared_conv.register_forward_hook(
                self._get_forward_hook('fixed_encoder.shared_conv')
            ))
        
        if hasattr(self.model.fixed_encoder, 'conv2'):
            self.hooks.append(self.model.fixed_encoder.conv2.register_forward_hook(
                self._get_forward_hook('fixed_encoder.conv2')
            ))
        
        if hasattr(self.model.fixed_encoder, 'conv3'):
            self.hooks.append(self.model.fixed_encoder.conv3.register_forward_hook(
                self._get_forward_hook('fixed_encoder.conv3')
            ))

        if hasattr(self.model.fixed_encoder, 'change_detector'):
            self.hooks.append(self.model.fixed_encoder.change_detector.register_forward_hook(
                self._get_forward_hook('fixed_encoder.change_detector')
            ))
        
        if hasattr(self.model.fusion, 'attention'):
            self.hooks.append(self.model.fusion.attention.register_forward_hook(
                self._get_forward_hook('fusion.attention')
            ))
        
        if hasattr(self.model.decoder, 'conv_block1'):
            self.hooks.append(self.model.decoder.conv_block1.register_forward_hook(
                self._get_forward_hook('decoder.conv_block1')
            ))
        
        if hasattr(self.model.decoder, 'conv_block2'):
            self.hooks.append(self.model.decoder.conv_block2.register_forward_hook(
                self._get_forward_hook('decoder.conv_block2')
            ))
        
        if hasattr(self.model.decoder, 'conv_block3'):
            self.hooks.append(self.model.decoder.conv_block3.register_forward_hook(
                self._get_forward_hook('decoder.conv_block3')
            ))
    
    def _get_forward_hook(self, name: str):
        """Create forward hook to save feature maps"""
        def hook(module, input, output):
            if name in TARGET_LAYERS:
                self.feature_maps[name] = output.detach()
        return hook
    
    def _generate_cam(self, feature_map: torch.Tensor, weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """Generate class activation map"""
        if len(feature_map.shape) != 4:
            if len(feature_map.shape) == 3:
                feature_map = feature_map.unsqueeze(0)
            elif len(feature_map.shape) == 5:
                feature_map = feature_map.squeeze(2)
        
        if weights is None:
            weights = torch.ones(feature_map.size(1), device=self.device) / feature_map.size(1)
        
        cam = torch.sum(weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * feature_map, dim=1)
        cam = cam.squeeze(0).cpu().numpy()
        
        cam = np.maximum(cam, 0)
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam
    
    def _upsample_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Upsample CAM to target size, ensuring 2D input"""
        if len(cam.shape) == 1:
            cam = np.expand_dims(cam, axis=0)
        elif len(cam.shape) > 2:
            cam = np.mean(cam, axis=0)
        
        if len(target_size) != 2:
            raise ValueError(f"Target size must be 2D tuple, got {target_size}")
            
        return cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    
    def visualize_sample(self, 
                        x_before: torch.Tensor, 
                        x_after: torch.Tensor, 
                        dynamic_inputs: List[torch.Tensor],
                        image: np.ndarray,
                        label: np.ndarray,
                        sample_name: str,
                        save_dir: str,
                        threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """Visualize CAM maps for a single sample (without generating overlay maps)"""
        x_before = x_before.to(self.device)
        x_after = x_after.to(self.device)
        dynamic_inputs = [inp.to(self.device) for inp in dynamic_inputs] if dynamic_inputs else []
        
        if len(x_before.shape) == 3:
            x_before = x_before.unsqueeze(0)
        if len(x_after.shape) == 3:
            x_after = x_after.unsqueeze(0)
        dynamic_inputs = [inp.unsqueeze(0) if len(inp.shape) == 3 else inp for inp in dynamic_inputs]
        
        with torch.no_grad():
            outputs = self.model(x_before, x_after, dynamic_inputs)
            preds = (torch.nn.functional.softmax(outputs, dim=1)[:, 1] > threshold).float()
        
        if len(image.shape) == 3:
            target_size = (image.shape[1], image.shape[0])
        elif len(image.shape) == 2:
            target_size = (image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Image must be 2D or 3D array, got shape {image.shape}")
        
        cam_results = {}
        for layer_name in TARGET_LAYERS:
            if layer_name in self.feature_maps:
                feature_map = self.feature_maps[layer_name]
                if len(feature_map.shape) in [3, 4, 5]:
                    cam = self._generate_cam(feature_map)
                    cam_upsampled = self._upsample_cam(cam, target_size)
                    cam_results[layer_name] = cam_upsampled
        
        sample_save_dir = os.path.join(save_dir, sample_name)
        os.makedirs(sample_save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(sample_save_dir, "original_image.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(label, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(sample_save_dir, "label.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(preds.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(sample_save_dir, "prediction.png"), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
        
        for layer_name, cam in cam_results.items():
            plt.figure(figsize=(10, 8))
            plt.imshow(cam, cmap='jet')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(sample_save_dir, f"cam_{layer_name}.png"), 
                       bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return cam_results
    
    def visualize_dataset(self, 
                         full_dataset,
                         sample_names: List[str],
                         feature_types: List[str],
                         feature_info: Dict[str, int],
                         save_dir: str,
                         num_samples: int = 10,
                         threshold: float = 0.5):
        """Visualize CAM maps for dataset samples - take specified number from the start"""
        os.makedirs(save_dir, exist_ok=True)
        print(f"Starting generation of CAM visualization results, saving to: {save_dir}")
        print(f"Using feature visualization method: {self.vis_method} - {VISUALIZATION_METHODS[self.vis_method]}")
        
        num_visualize = min(num_samples, len(full_dataset))
        print(f"Will select first {num_visualize} samples from complete dataset to generate CAM maps")
        
        for i in range(num_visualize):
            features, labels = full_dataset[i]
            sample_name = sample_names[i]
            
            x_before, x_after, dynamic_inputs = self._split_features(
                features.unsqueeze(0), feature_types, feature_info
            )
            
            image_np = self._get_visualizable_image(x_before.squeeze(0))
            
            self.visualize_sample(
                x_before.squeeze(0),
                x_after.squeeze(0),
                [d.squeeze(0) for d in dynamic_inputs],
                image_np,
                labels.numpy(),
                sample_name,
                os.path.join(save_dir, "cam_visualizations"),
                threshold
            )
            if (i + 1) % 5 == 0 or i + 1 == num_visualize:
                print(f"Processed {i + 1}/{num_visualize} samples")
        
        print(f"CAM visualization completed, generated visualization results for {num_visualize} samples")
    
    def _split_features(self, features, feature_types, feature_info):
        """Split features into fixed and dynamic branches"""
        feature_slices = {}
        current_idx = 0
        for feature in feature_types:
            channels = feature_info[feature]
            feature_slices[feature] = slice(current_idx, current_idx + channels)
            current_idx += channels
        
        x_before = features[:, feature_slices['before'], :, :]
        x_after = features[:, feature_slices['after'], :, :]
        
        dynamic_inputs = []
        for feature in feature_types:
            if feature not in ['before', 'after']:
                dynamic_inputs.append(features[:, feature_slices[feature], :, :])
        
        return x_before, x_after, dynamic_inputs
    
    def _get_visualizable_image(self, x_before: torch.Tensor) -> np.ndarray:
        """Extract visualizable image from features with multiple visualization methods"""
        if len(x_before.shape) == 4:
            x_before = x_before.squeeze(0)
        
        features_np = x_before.cpu().numpy()
        num_channels = features_np.shape[0]
        
        if self.vis_method == 'first_three' or num_channels <= 3:
            if num_channels >= 3:
                image_data = features_np[:3, :, :]
            else:
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
                
        elif self.vis_method == 'max_variance':
            if num_channels <= 3:
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
            else:
                variances = np.var(features_np, axis=(1, 2))
                top_indices = np.argsort(variances)[-3:][::-1]
                image_data = features_np[top_indices, :, :]
                
        elif self.vis_method == 'pca':
            if num_channels <= 3:
                image_data = np.repeat(features_np, 3, axis=0)[:3, :, :]
            else:
                h, w = features_np.shape[1], features_np.shape[2]
                flattened = features_np.reshape(num_channels, -1)
                flattened = flattened - np.mean(flattened, axis=1, keepdims=True)
                cov_matrix = np.cov(flattened)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                top_indices = np.argsort(eigenvalues)[-3:][::-1]
                top_eigenvectors = eigenvectors[:, top_indices]
                pca_result = np.dot(top_eigenvectors.T, flattened)
                image_data = pca_result.reshape(3, h, w)
                
        elif self.vis_method == 'grayscale_mean':
            mean_channel = np.mean(features_np, axis=0, keepdims=True)
            image_data = np.repeat(mean_channel, 3, axis=0)
            
        elif self.vis_method == 'grayscale_max':
            max_channel = np.max(features_np, axis=0, keepdims=True)
            image_data = np.repeat(max_channel, 3, axis=0)
        
        image = np.transpose(image_data, (1, 2, 0))
        
        for i in range(3):
            min_val = np.min(image[:, :, i])
            max_val = np.max(image[:, :, i])
            if max_val > min_val:
                image[:, :, i] = (image[:, :, i] - min_val) / (max_val - min_val) * 255
        
        return image.astype(np.uint8)
    
    def __del__(self):
        """Clean up hooks"""
        for hook in self.hooks:
            hook.remove()


def generate_cam_from_weights(config):
    """Generate CAM visualization results from pre-trained weights"""
    logger = setup_logger(config['log_dir'])
    logger.info("===== Generating CAM Visualizations from Pre-trained Weights =====")
    logger.info(f"Using weight file: {config['weight_path']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count()
    
    if device_count < 2:
        logger.warning(f"Detected {device_count} GPU(s), while dual GPUs were used for training. Attempting to continue, but issues may occur.")
    else:
        logger.info(f"Using dual GPU mode, device: {device}")
    
    multiprocessing.set_start_method('spawn', force=True)
    
    _, _, _, feature_info, feature_types, full_dataset = create_dataloaders(config)
    sample_names = full_dataset.sample_names
    
    try:
        LandslideNet = dynamic_import_model(config['model_path'], config['model_name'])
        logger.info(f"Successfully imported model: {config['model_name']} from {config['model_path']}")
    except Exception as e:
        logger.error(f"Failed to import model: {str(e)}")
        raise
    
    dynamic_branch_channels = [
        feature_info[feature] for feature in feature_types if feature not in ['before', 'after']
    ]
    logger.info(f"Dynamic feature branch channels: {dynamic_branch_channels}")
    
    fixed_in_channels = feature_info['before']
    model = LandslideNet(
        fixed_in_channels=fixed_in_channels,
        dynamic_in_channels=dynamic_branch_channels
    )
    
    try:
        state_dict = torch.load(config['weight_path'], map_location='cpu')
        
        if device_count >= 2 and torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        
        model = model.to(device)
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded weight file: {config['weight_path']}")
    except Exception as e:
        logger.error(f"Failed to load weight file: {str(e)}")
        try:
            logger.info("Attempting to load after removing 'module.' prefix from weights...")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            logger.info("Successfully loaded weights after prefix removal")
        except Exception as e2:
            logger.error(f"Still failed after prefix removal: {str(e2)}")
            raise
    
    model.eval()
    
    vis_method = config.get('visualization_method', DEFAULT_VISUALIZATION)
    
    cam_visualizer = CAMVisualizer(
        model=model,
        feature_names=feature_types,
        device=device,
        vis_method=vis_method
    )
    
    cam_save_dir = os.path.join(config['output_dir'], 'cam_visualizations')
    os.makedirs(cam_save_dir, exist_ok=True)
    logger.info(f"CAM visualization results will be saved to: {cam_save_dir}")
    
    cam_visualizer.visualize_dataset(
        full_dataset=full_dataset,
        sample_names=sample_names,
        feature_types=feature_types,
        feature_info=feature_info,
        save_dir=cam_save_dir,
        num_samples=config.get('cam_samples', 10),
        threshold=config['pred_threshold']
    )
    
    return cam_save_dir


def main(config_path):
    """Main function"""
    config = parse_config(config_path)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    if not os.path.exists(config['weight_path']):
        raise FileNotFoundError(f"Weight file not found: {config['weight_path']}")
    
    cam_dir = generate_cam_from_weights(config)
    print(f"CAM visualization completed, results saved in: {cam_dir}")
    return cam_dir


if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise RuntimeError("Please provide configuration file path as an argument")
        
        config_path = sys.argv[1]
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        main(config_path)
        print('<status>0</status>')
        print('<message>CAM visualization generated successfully</message>')
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('\t', ' ')
        print('<status>1</status>')
        print(f'<message>{error_msg}</message>')
        logger = logging.getLogger('cam_generator')
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler())
        logger.error(f"Processing failed: {error_msg}")
        
