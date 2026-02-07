
import torch
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional
import math

class TransformerVisualizer:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hook_manager = None

    def _get_layer_names(self):
        """Helper to get transformer layer names. Attempts to be model-agnostic."""
        print("Model Architecture Scan:")
        layers = []
        for name, module in self.model.named_modules():
            # Heuristic for finding transformer blocks based on common naming conventions
            if any(key in name for key in ['h.', 'layers.', 'block.']) and name.count('.') == 2:
                # We want the output of the block essentially
                # Check if it's a ModuleList index
                try:
                    int(name.split('.')[-1])
                    layers.append(name)
                except ValueError:
                    continue
        
        # Sort layers by index
        layers.sort(key=lambda x: int(x.split('.')[-1]))
        return layers

    def visualize(self, text_input: str, top_k: int = 10):
        """
        Visualizes the activations for the given input text.
        """
        self.model.eval()
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        
        from .hook_manager import HookManager
        self.hook_manager = HookManager(self.model)
        
        # Identify layers to hook
        layer_names = self._get_layer_names()
        print(f"Hooking layers: {layer_names}")
        self.hook_manager.register_hooks(layer_names)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        activations = self.hook_manager.activations
        self.hook_manager.remove_hooks()
        
        return self._create_plot(activations, layer_names, top_k, text_input)

    def _create_plot(self, activations, layer_names, top_k, input_text):
        fig = go.Figure()

        # Configuration for visualization
        layer_spacing = 5.0
        neuron_spread = 2.0  # Spread on the layer surface
        
        feature_dim = list(activations.values())[0].shape[-1]
        
        # We need a 2D mapping for neurons. 
        # Simple approach: Map 1D index to 2D grid
        grid_size = int(math.ceil(math.sqrt(feature_dim)))
        
        # Pre-compute grid positions for neurons
        neuron_pos = {}
        for i in range(feature_dim):
            r = i // grid_size
            c = i % grid_size
            # Center the grid
            y = (c - grid_size / 2) * (neuron_spread / grid_size)
            z = (r - grid_size / 2) * (neuron_spread / grid_size)
            neuron_pos[i] = (y, z)

        # Store top-k indices per layer to draw connections
        layer_top_indices = []

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name not in activations:
                # Fallback for unexpected missing hooks
                print(f"Warning: No activation found for {layer_name}")
                layer_top_indices.append([]) 
                continue

            act = activations[layer_name]
            # act shape: [batch, seq, features]
            # Take last token: [features]
            last_token_act = act[0, -1, :] 
            
            # Find top-k
            values, indices = torch.topk(last_token_act, k=top_k)
            indices = indices.tolist()
            values = values.tolist()
            
            layer_top_indices.append(indices)
            
            # X-coordinate for this layer (Lateral flow)
            x = layer_idx * layer_spacing
            
            # Add layer surface (plane)
            # Create a mesh for the plane (oriented on YZ plane at specific X)
            plane_range = 1.5
            fig.add_trace(go.Mesh3d(
                x=[x, x, x, x],
                y=[-plane_range, plane_range, plane_range, -plane_range],
                z=[-plane_range, -plane_range, plane_range, plane_range],
                color='rgba(200, 200, 255, 0.1)',
                hoverinfo='skip'
            ))

            # Add neurons (Scatter3d)
            # Only visualizing Top-K active neurons to keep it clean, as requested
            
            l_x, l_y, l_z, l_c, l_t = [], [], [], [], []
            
            for idx, val in zip(indices, values):
                ny, nz = neuron_pos[idx]
                l_x.append(x)
                l_y.append(ny)
                l_z.append(nz)
                l_c.append(val)
                l_t.append(f"Layer: {layer_idx}<br>Neuron: {idx}<br>Value: {val:.4f}")
            
            fig.add_trace(go.Scatter3d(
                x=l_x, y=l_y, z=l_z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=l_c,
                    colorscale='Viridis',
                    opacity=0.9
                ),
                text=l_t,
                name=f"Layer {layer_idx}"
            ))

        # Add connections between layers
        
        edge_x, edge_y, edge_z = [], [], []
        
        for i in range(len(layer_names) - 1):
            source_indices = layer_top_indices[i]
            target_indices = layer_top_indices[i+1]
            
            x_start = i * layer_spacing
            x_end = (i + 1) * layer_spacing
            
            for s_idx in source_indices:
                sy, sz = neuron_pos[s_idx]
                
                # SPARSE CONNECTION LOGIC:
                # We only connect this source neuron to a subset of target neurons
                # For visualization, let's just connect to the top-2 strongest target neurons
                # regardless of spatial position, to show "strongest signal paths".
                
                # target_indices are already sorted by activation strength from topk
                targets_to_connect = target_indices[:2] # Top 2
                
                for t_idx in targets_to_connect:
                    ty, tz = neuron_pos[t_idx]
                    
                    edge_x.extend([x_start, x_end, None])
                    edge_y.extend([sy, ty, None])
                    edge_z.extend([sz, tz, None])

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(255, 100, 100, 0.3)', width=2),
            hoverinfo='skip',
            name='Flow'
        ))

        fig.update_layout(
            title=f"Transformer Activation Flow: '{input_text}'",
            scene=dict(
                xaxis_title="Layer Depth",
                yaxis_title="Neuron Grid Y",
                zaxis_title="Neuron Grid Z",
                aspectmode='manual',
                aspectratio=dict(x=3, y=1, z=1) # Elongated along X for lateral view
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
