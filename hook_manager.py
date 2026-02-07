
import torch
from typing import Dict, List, Optional, Tuple

class HookManager:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def _get_activation_hook(self, name):
        def hook(model, input, output):
            # Handle different output types (tuple, tensor, etc.)
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Detach and move to CPU to save memory
            self.activations[name] = activation.detach().cpu()
        return hook

    def register_hooks(self, layer_names: List[str]):
        """
        Registers hooks on the specified layers.
        
        Args:
            layer_names: List of module names to hook into. 
                         e.g. ['transformer.h.0', 'transformer.h.1'] for GPT-2
        """
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations
