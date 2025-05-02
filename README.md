# ComfyUI-VideoUpscale_WithModel

![image](https://github.com/user-attachments/assets/1abcca90-60dc-480c-a960-e8068e417bdb)


A memory-efficient implementation for upscaling videos in ComfyUI using non-diffusion upscaling models. This custom node is designed to handle large video frame sequences without memory bottlenecks.

## Overview

This node brings the power of non-diffusion upscaling models like ESRGAN to video processing in ComfyUI. While these models have long been the gold standard for image upscaling, applying them to video has been challenging due to memory constraints. This implementation solves those challenges with smart memory management strategies.

## Features

- **Memory-Efficient Processing**: Three different memory management strategies to fit your hardware
- **Green Progress Bar**: Real-time terminal progress tracking with ETA
- **Integrated Model Loading**: Direct model selection without needing separate loader nodes
- **High-Quality Results**: Leverages the detail-preserving capabilities of non-diffusion models
- **Simple Workflow**: Just connect video frames and select your model

## Why Non-Diffusion Upscalers?

Non-diffusion upscalers like ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) and its variants offer significant advantages for video processing:

- **Speed**: Process frames much faster than diffusion-based methods
- **Memory Efficiency**: Require less VRAM per frame than diffusion models
- **Detail Preservation**: Excellent at preserving fine details and textures
- **Artifact Reduction**: Reduce common video artifacts like compression noise
- **Domain Specialization**: Models trained for specific content types (anime, realistic photos, etc.)

Unlike diffusion-based upscalers that generate entirely new content, non-diffusion models like ESRGAN focus on enhancing what's already there, making them perfect for video where consistency between frames is critical.

## The OpenModelDB Connection

This node works wonderfully with models from [OpenModelDB](https://openmodeldb.info/), a community-driven database of specialized AI upscaling models. OpenModelDB hosts hundreds of models trained for specific purposes:

- Video frame enhancement
- Anime/cartoon upscaling
- Realistic photo improvement
- Game texture enhancement
- Denoising and compression artifact removal
- And many more specialized tasks

Simply download models from OpenModelDB and place them in your ComfyUI upscale_models folder to use them with this node.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ShmuelRonen/ComfyUI-VideoUpscale_WithModel
```

## Usage

1. Place videos frames in the "images" input
2. Select your preferred upscale model
3. Choose an upscaling method and factor
4. Select a memory management strategy:
   - `auto`: Automatically selects the best strategy based on available VRAM
   - `keep_loaded`: Keeps model on GPU for maximum speed (uses most VRAM)
   - `load_unload_each_frame`: Loads/unloads model between frames (balanced approach)
   - `cpu_only`: Processes everything on CPU (minimal VRAM usage)

## Memory Management Strategies Explained

### keep_loaded

This strategy keeps the upscale model on the GPU for the entire processing duration. It's the fastest option but uses the most VRAM. Best for high-end GPUs with plenty of memory.

### load_unload_each_frame

This balanced approach loads the model to GPU for processing each frame, then moves it back to CPU. It offers a good compromise between speed and memory usage, ideal for mid-range GPUs.

### cpu_only

This strategy processes everything on the CPU without using GPU memory. It's the slowest option but uses minimal VRAM, making it suitable for systems with limited GPU resources or CPU-only setups.

## Example Workflow

```
Load Video → Video_Upscale_With_Model → Free_Video_Memory → VAE Decode → Save Video
```

## Performance Tips

- **Batch Processing**: This node intentionally processes one frame at a time to minimize memory usage, making it ideal for long videos.
- **Tile Size**: Different memory strategies use different tile sizes to balance memory usage and speed.
- **Memory Cleanup**: The companion `Free_Video_Memory` node can be placed after this node to ensure thorough memory cleanup.

## License

MIT License

## Acknowledgments

- The ComfyUI team for their amazing framework
- The creators of ESRGAN and other non-diffusion upscaling models
- The [OpenModelDB](https://openmodeldb.info/) community for their specialized upscale models
