
my visualization exeeriments: Transformer Activation Visualizer

here is a  lightweight Python library for visualizing the internal activation flows of Transformer models. It is designed to work with Hugging Face `transformers` and supports quantized models (via `bitsandbytes`).

cool sfeatures

3D Visualization: Visualizes transformer layers as parallel surfaces in a 3D space.
Neuron Activations: plots the top-k most active neurons for a given input.
Flow Visualization: Connects active neurons across layers to show the "thought process" of the model.
Quantization Support: Seamlessly works with 4-bit/8-bit quantized models using `bitsandbytes`.
Interactive**: Generates interactive Plotly figures.


See `demo_notebook.ipynb` for a complete example.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from intra_viz import TransformerVisualizer

# Load Model
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Visualize
viz = TransformerVisualizer(model, tokenizer)
fig = viz.visualize("The quick brown fox jumps over the lazy dog", top_k=10)
fig.show()
```
Simple!!


How it Works

The library works by registering forward hooks on the model's transformer layers.
It runs a single forward pass with the provided input text.
Activations are captured and moved to CPU.
Plotting:
    *   Neurons are mapped to a 2D grid on each layer's surface.
    *   Surfaces are stacked along the Z-axis (depth).
    *   Top-K activations are highlighted and connected.
