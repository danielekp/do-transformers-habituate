import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class MLPActivationRecorder:
    """Custom hook system to record projection MLP activations from any transformer model."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str) -> None:
        """Initialize with a model and a tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Storage for activations
        self.activations = defaultdict(list)
        self.hooks = []
        self.current_token_pos = None
        
        print(f"Recorder loaded successfully!")
        print(f"Number of layers: {self.model.config.max_window_layers if hasattr(self.model.config, 'max_window_layers') else 'Unknown'}")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        
    def mlp_hook_factory(self, layer_idx: int, token_positions: List[int]):
        """Factory function to create hooks for MLP outputs at specific token positions."""
        def hook_fn(module, input, output):
            """Hook function that captures MLP output for specified token positions."""
            # output shape: [batch_size, seq_len, hidden_size]
            if isinstance(output, tuple):
                output = output[0]  # Take first element if tuple
                
            batch_size, seq_len, hidden_size = output.shape
            
            # Store activations for each target token position
            for pos in token_positions:
                if pos < seq_len:  # Make sure position is valid
                    activation = output[0, pos, :].detach().cpu().clone()  # [hidden_size]
                    self.activations[f'layer_{layer_idx}_pos_{pos}'].append({
                        'activation': activation,
                        'magnitude': torch.norm(activation).item(),
                        'layer': layer_idx,
                        'position': pos
                    })
        
        return hook_fn
    
    def register_mlp_hooks(self, token_positions: List[int]):
        """Register hooks on all MLP layers to capture activations."""
        self.clear_hooks()
        
        print(f"Registering hooks for token positions: {token_positions}")
        
        # Different model architectures have different ways to access MLP layers
        if hasattr(self.model, 'transformer'):  # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):  # Llama/Gemma style
            layers = self.model.model.layers
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):  # GPT-NeoX style
            layers = self.model.gpt_neox.layers
        else:
            raise ValueError("Unknown model architecture - cannot locate MLP layers")
        
        hooks_registered = 0
        for layer_idx, layer in enumerate(layers):
            # Try to find MLP module in the layer
            mlp_module = None
            
            # Common MLP module names
            if hasattr(layer, 'mlp'):
                mlp_module = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_module = layer.feed_forward  
            elif hasattr(layer, 'ffn'):
                mlp_module = layer.ffn
            
            if mlp_module is not None:
                hook = mlp_module.register_forward_hook(
                    self.mlp_hook_factory(layer_idx, token_positions)
                )
                self.hooks.append(hook)
                hooks_registered += 1
        
        print(f"Registered {hooks_registered} MLP hooks")
        return hooks_registered
    
    def tokenize_and_find_positions(self, text: str, target_token: str) -> Tuple[torch.Tensor, List[int]]:
        """Tokenize text and find all positions of target token."""
        # Tokenize
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        token_strings = self.tokenizer.convert_ids_to_tokens(tokens[0])
        
        print(f"Text: {text}")
        
        # Find target token positions
        target_positions = []
        for i, token_str in enumerate(token_strings):
            # Handle subword tokenization (tokens might have special prefixes)
            clean_token = token_str.lower().replace('▁', '').replace('Ġ', '').replace('##', '')
            if target_token.lower() in clean_token:
                target_positions.append(i)
        
        if not target_positions:
            print(f"Warning: '{target_token}' not found. Available tokens: {token_strings}")
        
        return tokens, target_positions
    
    def run_inference_with_hooks(self, text: str, target_token: str) -> Dict:
        """Run inference while recording MLP activations for target token."""
        # Clear previous activations
        self.activations.clear()
        
        # Tokenize and find target positions
        tokens, target_positions = self.tokenize_and_find_positions(text, target_token)
        
        if not target_positions:
            return {'error': 'Target token not found'}
        
        # Register hooks
        num_hooks = self.register_mlp_hooks(target_positions)
        
        # Run inference
        print(f"Running inference on {tokens.shape[1]} tokens...")
        with torch.no_grad():
            outputs = self.model(tokens)
        
        # Process results
        results = {
            'tokens': tokens,
            'target_positions': target_positions,
            'num_layers_hooked': num_hooks,
            'activations': dict(self.activations)  # Convert defaultdict to regular dict
        }
        
        print(f"Captured activations for {len(self.activations)} position-layer combinations")
        return results
    
    def find_best_layer_for_token(self, text: str, target_token: str) -> Tuple[int, float]:
        """Find which layer produces highest magnitude activation for target token."""
        results = self.run_inference_with_hooks(text, target_token)
        
        if 'error' in results:
            raise ValueError(results['error'])
        
        # Find the first occurrence of target token
        first_target_pos = results['target_positions'][0]
        
        # Compare magnitudes across layers for this position
        layer_magnitudes = []
        best_layer = 0
        max_magnitude = 0
        
        for key, activations in results['activations'].items():
            if f'pos_{first_target_pos}' in key:
                layer_idx = int(key.split('layer_')[1].split('_pos')[0])
                magnitude = activations[0]['magnitude']  # First (and only) activation
                layer_magnitudes.append((layer_idx, magnitude))
                
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    best_layer = layer_idx
        
        # Sort by layer index for plotting
        layer_magnitudes.sort(key=lambda x: x[0])
        
        # Plot results
        if layer_magnitudes:
            layers, magnitudes = zip(*layer_magnitudes)
            plt.figure(figsize=(12, 6))
            plt.plot(layers, magnitudes, 'o-', linewidth=2, markersize=8)
            plt.axvline(x=best_layer, color='red', linestyle='--', alpha=0.7, 
                       label=f'Best layer: {best_layer} (mag: {max_magnitude:.4f})')
            plt.xlabel('Layer Index')
            plt.ylabel('MLP Output Magnitude')
            plt.title(f'MLP Output Magnitudes Across Layers for "{target_token}"')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        print(f"Best layer: {best_layer} with magnitude: {max_magnitude:.4f}")
        return best_layer, max_magnitude