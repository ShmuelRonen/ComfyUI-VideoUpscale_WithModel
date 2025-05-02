from .video_upscale_model import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Print initialization message for debugging
print("Initializing ComfyUI-VideoUpscale_WithModel")

# Try to initialize the progress handling for ComfyUI
try:
    from .video_upscale_model import init_comfyui_progress
    init_comfyui_progress()
except Exception as e:
    print(f"Note: Progress bar initialization failed: {e}")

# Export node mappings for ComfyUI to detect
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]