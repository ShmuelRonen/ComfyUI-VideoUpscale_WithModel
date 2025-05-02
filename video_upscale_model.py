import os
import torch
import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import gc
import time

class Video_Upscale_With_Model:
    """
    A memory-efficient implementation for upscaling video frames using an upscale model
    with proper batch processing and memory management.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                    "images": ("IMAGE",),
                    "upscale_method": (s.upscale_methods,),
                    "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                    "device_strategy": (["auto", "load_unload_each_frame", "keep_loaded", "cpu_only"], {"default": "auto"})
                }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_video"
    CATEGORY = "video"
    
    # ComfyUI progress bar integration
    def __init__(self):
        self.steps = 0
        self.step = 0
    
    # Important: ComfyUI looks for this method to track progress
    def get_progress_execution(self):
        if self.steps > 0:
            return self.step, self.steps
        return 0, 1
    
    def upscale_video(self, model_name, images, upscale_method, factor, device_strategy="auto"):
        """
        Upscale a sequence of images (video frames) efficiently using an integrated upscale model.
        """
        # Load the upscale model
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)
        upscale_model = self.load_upscale_model(upscale_model_path)
        
        # Determine the right strategy
        device = model_management.get_torch_device()
        if device_strategy == "auto":
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                
                if (total_memory - reserved_memory) / total_memory > 0.5:
                    device_strategy = "keep_loaded"
                else:
                    device_strategy = "load_unload_each_frame"
            else:
                device_strategy = "cpu_only"
        
        # Get dimensions
        num_frames = images.shape[0]
        old_height = images.shape[1]
        old_width = images.shape[2]
        new_height = int(old_height * factor)
        new_width = int(old_width * factor)
        
        # Initialize progress tracking
        self.steps = num_frames
        self.step = 0
        
        print(f"Processing video: {num_frames} frames from {old_width}x{old_height} to {new_width}x{new_height} with {device_strategy} strategy")
        
        # Strategy-based upscaling (always batch_size of 1 for simplicity)
        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result_frames = self._upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height)
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result_frames = self._upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height)
        else:  # "load_unload_each_frame"
            result_frames = self._upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height)
        
        # Stack frames back into a single tensor
        return (torch.stack(result_frames),)
    
    def load_upscale_model(self, model_path):
        """Load the upscale model from the given path"""
        from comfy_extras.chainner_models import model_loading
        
        sd = comfy.utils.load_torch_file(model_path)
        upscale_model = model_loading.load_state_dict(sd).eval()
        
        # Free up memory
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return upscale_model
    
    def _upscale_on_cpu(self, upscale_model, images, upscale_method, new_width, new_height):
        """Process all frames on CPU to minimize VRAM usage"""
        result_frames = []
        start_time = time.time()
        
        # Process frames one by one
        for i in range(images.shape[0]):
            # Get the current frame
            frame = images[i:i+1]  # Keep batch dimension
            
            # Convert frame for upscaling model
            in_img = frame.movedim(-1, -3)  # [B, H, W, C] -> [B, C, H, W]
            
            # Apply the upscaling model
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=64,  # Smaller tiles to reduce memory usage
                tile_y=64,
                overlap=8, 
                upscale_amount=upscale_model.scale
            )
            
            # Convert and resize
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # [B, C, H, W] -> [B, H, W, C]
            samples = upscaled.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s = s.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]
            
            # Add to results
            result_frames.append(s[0])  # Remove batch dimension
            
            # Update progress
            self.step += 1
            
            # Calculate ETA and print progress
            elapsed = time.time() - start_time
            if self.step > 0:
                eta = elapsed / self.step * (self.steps - self.step)
            else:
                eta = 0
            
            # Format elapsed and ETA times
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            # Clean up
            del in_img, s, upscaled, samples
            gc.collect()
        
        print()  # Final newline
        return result_frames
    
    def _upscale_batch_keep_loaded(self, upscale_model, images, device, upscale_method, new_width, new_height):
        """Keep model on GPU for entire processing (highest VRAM usage but fastest)"""
        result_frames = []
        start_time = time.time()
        
        # Process frames one by one
        for i in range(images.shape[0]):
            # Get the current frame
            frame = images[i:i+1]  # Keep batch dimension
            
            # Convert frame for upscaling model
            in_img = frame.movedim(-1, -3).to(device)  # [B, H, W, C] -> [B, C, H, W]
            
            # Apply the upscaling model (already on device)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=128,  # Larger tiles since we have VRAM to spare
                tile_y=128,
                overlap=8, 
                upscale_amount=upscale_model.scale
            )
            
            # Convert and resize
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # [B, C, H, W] -> [B, H, W, C]
            samples = upscaled.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s = s.movedim(1, -1).cpu()  # [B, C, H, W] -> [B, H, W, C]
            
            # Add to results
            result_frames.append(s[0])  # Remove batch dimension
            
            # Update progress
            self.step += 1
            
            # Calculate ETA and print progress
            elapsed = time.time() - start_time
            if self.step > 0:
                eta = elapsed / self.step * (self.steps - self.step)
            else:
                eta = 0
            
            # Format elapsed and ETA times
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            # Clean up
            del in_img, s, upscaled, samples
            torch.cuda.empty_cache()
        
        print()  # Final newline
        return result_frames
    
    def _upscale_batch_load_unload(self, upscale_model, images, device, upscale_method, new_width, new_height):
        """Load model to GPU for each frame batch, then move back to CPU (balanced approach)"""
        result_frames = []
        start_time = time.time()
        
        # Process frames one by one with model loading/unloading
        for i in range(images.shape[0]):
            # Clear cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model to GPU for this frame
            upscale_model = upscale_model.to(device)
            
            # Get the current frame
            frame = images[i:i+1]  # Keep batch dimension
            
            # Convert frame for upscaling model
            in_img = frame.movedim(-1, -3).to(device)  # [B, H, W, C] -> [B, C, H, W]
            
            # Apply the upscaling model
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=96,  # Medium-sized tiles for balance
                tile_y=96,
                overlap=8, 
                upscale_amount=upscale_model.scale
            )
            
            # Convert and resize
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)  # [B, C, H, W] -> [B, H, W, C]
            samples = upscaled.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            s = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s = s.movedim(1, -1).cpu()  # [B, C, H, W] -> [B, H, W, C]
            
            # Add to results
            result_frames.append(s[0])  # Remove batch dimension
            
            # Update progress
            self.step += 1
            
            # Calculate ETA and print progress
            elapsed = time.time() - start_time
            if self.step > 0:
                eta = elapsed / self.step * (self.steps - self.step)
            else:
                eta = 0
            
            # Format elapsed and ETA times
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            # Clean up
            del in_img, s, upscaled, samples
            
            # Move model back to CPU after processing this frame
            upscale_model = upscale_model.to("cpu")
            
            # Aggressively clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()  # Final newline
        return result_frames


# Optional companion node to manage memory explicitly during video processing
class Free_Video_Memory:
    """
    A node that explicitly cleans up memory during video processing pipelines
    to avoid memory bottlenecks.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "aggressive_cleanup": (["disable", "enable"], {"default": "disable"}),
            "report_memory": (["disable", "enable"], {"default": "enable"})
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cleanup_memory"
    CATEGORY = "video"
    
    def cleanup_memory(self, images, aggressive_cleanup="disable", report_memory="enable"):
        # Report memory usage before cleanup
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"Before cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Run garbage collection
        gc.collect()
        
        # Handle CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # More aggressive memory cleanup
            if aggressive_cleanup == "enable":
                # Force a synchronization point
                torch.cuda.synchronize()
                
                # Try to defragment memory
                if hasattr(torch.cuda, 'caching_allocator_delete_caches'):
                    torch.cuda.caching_allocator_delete_caches()
        
        # Report memory usage after cleanup
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Return the images unmodified - this node is just for memory management
        return (images,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Video_Upscale_With_Model": Video_Upscale_With_Model,
    "Free_Video_Memory": Free_Video_Memory,
}

# Display name and category mappings for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "Video_Upscale_With_Model": "Video Upscale With Model",
    "Free_Video_Memory": "Free Video Memory",
}