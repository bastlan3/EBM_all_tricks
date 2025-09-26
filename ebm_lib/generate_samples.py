import torch
from typing import Optional, Union
from ebm_lib.ebm import EBM
import os
from ebm_lib.main import SimpleCNN  # Import the actual model from main
from ebm_lib.samplers.langevin import LangevinSampler  # Add this import
import torchvision.utils as vutils
import torch.nn as nn

def generate_samples(
    checkpoint_path: str,
    model_class: nn.Module,
    sampler,
    num_samples: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_shape: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None
) -> torch.Tensor:
    """
    Load a checkpoint and generate samples using a given sampler.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_class: The model class to instantiate
        sampler: Sampling method (e.g., Langevin, MCMC sampler)
        num_samples: Number of samples to generate
        device: Device to run on
        sample_shape: Shape of samples (if not inferred from model)
        model_kwargs: Additional arguments for model initialization
    
    Returns:
        Generated samples as torch.Tensor
    """
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with proper parameters
    model_kwargs = model_kwargs or {}
    
    # Try to get model parameters from checkpoint if available
    if 'model_config' in checkpoint:
        model_kwargs.update(checkpoint['model_config'])
    
    # Determine the correct key for model state dict
    state_dict_key = None
    possible_keys = ['model_state_dict', 'state_dict', 'model', 'net']
    for key in possible_keys:
        if key in checkpoint:
            state_dict_key = key
            break
    
    if state_dict_key is None:
        # If no standard key found, assume the checkpoint IS the state dict
        if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise KeyError(f"Could not find model state dict in checkpoint. Available keys: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint[state_dict_key]
    
    # Handle wrapped models (e.g., EBM wrapper)
    def extract_model_state_dict(state_dict):
        """Extract the actual model state dict from potentially wrapped models."""
        # Check if keys have a common prefix that needs to be removed
        keys = list(state_dict.keys())
        
        # Look specifically for the network weights within EBM wrapper
        network_prefix = 'ebm_model.network.'
        if any(key.startswith(network_prefix) for key in keys):
            # Extract only the network weights, ignore other EBM components
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(network_prefix):
                    new_key = key[len(network_prefix):]
                    new_state_dict[new_key] = value
            return new_state_dict
        
        # Fallback to general prefix handling
        prefixes = ['model.', 'net.', 'network.']
        for prefix in prefixes:
            if any(key.startswith(prefix) for key in keys):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        new_state_dict[new_key] = value
                return new_state_dict
        
        return state_dict
    
    # Extract the actual model weights
    clean_state_dict = extract_model_state_dict(state_dict)
    
    try:
        model = model_class(**model_kwargs)
        model.load_state_dict(clean_state_dict)
    except Exception as e:
        # Fallback: try with required arguments
        print(f"Warning: Failed to load model with config {model_kwargs}. Trying with minimal required args. Error: {e}")
        try:
            # For SimpleCNN, input_shape is required
            if hasattr(model_class, '__init__') and 'input_shape' not in model_kwargs:
                model_kwargs['input_shape'] = (3, 32, 32)  # Default CIFAR-10 shape
            model = model_class(**model_kwargs)
            model.load_state_dict(clean_state_dict)
        except Exception as e2:
            # Debug: show what keys are available vs expected
            print(f"Available state dict keys: {list(clean_state_dict.keys())}")
            model_temp = model_class(**model_kwargs) if model_kwargs else model_class(input_shape=(3, 32, 32))
            print(f"Expected model keys: {list(model_temp.state_dict().keys())}")
            raise RuntimeError(f"Failed to load model: {e2}")
    
    model = EBM(model)  # Wrap the model in EBM
    model.eval()
    
    samples = sampler.sample(
        ebm_model=model,
        n_samples=num_samples,
        sample_shape=sample_shape or (3, 32, 32)
    )
    
    return samples

def main():
    # Example usage
    checkpoint_path = "/home/bastienll/Documents/PHD/EBM_toy_models/EBM_all_tricks/checkpoints/epoch=9-step=7820.ckpt"
    
    # Model configuration
    model_config = {
        'input_shape': (3, 32, 32)
    }
    
    sampler_config = {
        'k_steps': 40,
        'step_size': 0.01,
        'noise_scale': 0.01,
        'clip_grad': True,
        'clip_value': 0.03
    }
    sampler = LangevinSampler(config=sampler_config)

    samples = generate_samples(
        checkpoint_path=checkpoint_path,
        model_class=SimpleCNN,  # Pass the class, not an instance
        sampler=sampler,
        num_samples=10,
        model_kwargs=model_config  # Pass model configuration as kwargs
    )

    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)
    vutils.save_image(samples, os.path.join(output_dir, "samples.png"), nrow=5, normalize=True)
    print(f"Samples saved to {os.path.join(output_dir, 'samples.png')}")

    pass

if __name__ == "__main__":
    main()