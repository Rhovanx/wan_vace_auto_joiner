"""
WAN VACE Auto Joiner - For Loop Compatible System

Three nodes designed to work with ComfyUI-Easy-Use For Loop:

1. WanVaceAutoJoiner - Handles INIT (index=1) and PROCESS (index>1)
2. WanVaceAutoJoinerSave - Saves VACE output to disk (inside loop)
3. WanVaceAutoJoinerFinalize - Outputs final video (after loop, NO VACE!)

For N videos (N-1 transitions):
- Set For Loop Start total = N-1
- Loop runs N-1 times with VACE processing
- Finalize runs ONCE after loop with NO VACE overhead!

Uses 1-based indexing internally (For Loop's 0-based index + 1).
"""

import os
import json
import glob
import shutil
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import torch
import numpy as np
from PIL import Image
import cv2


# =============================================================================
# Shared Constants
# =============================================================================

OVERLAP_FRAMES = 16
NEXT_FRAMES = 17
TOTAL_BATCH = 33
MASK_START = 8
MASK_END = 24


# =============================================================================
# Shared Helper Functions
# =============================================================================

def get_video_path(directory: str, prefix: str, suffix: int) -> str:
    filename = f"{prefix}_{suffix:05d}.mp4"
    return os.path.join(directory, filename)


def get_frame_path(temp_folder: str, prefix: str, frame_num: int) -> str:
    filename = f"{prefix}_{frame_num:05d}.png"
    return os.path.join(temp_folder, filename)


def find_temp_folder(directory: str) -> Optional[str]:
    pattern = os.path.join(directory, "temp-*")
    temp_folders = glob.glob(pattern)
    
    for folder in sorted(temp_folders, reverse=True):
        state_file = os.path.join(folder, "state.json")
        if os.path.exists(state_file):
            return folder
    return None


def create_temp_folder(directory: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    temp_folder = os.path.join(directory, f"temp-{timestamp}")
    os.makedirs(temp_folder, exist_ok=True)
    return temp_folder


def load_state(temp_folder: str) -> Optional[Dict[str, Any]]:
    state_file = os.path.join(temp_folder, "state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return None


def save_state(temp_folder: str, state: Dict[str, Any]) -> None:
    state_file = os.path.join(temp_folder, "state.json")
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return width, height, frame_count, fps


def read_video_frames(video_path: str, start: int = 0, 
                      end: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if end is None:
        end = frame_count
    end = min(end, frame_count)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames


def create_solid_image(width: int, height: int, 
                       color: Tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def save_frame(frame: np.ndarray, path: str) -> None:
    img = Image.fromarray(frame)
    img.save(path)


def save_frames_to_temp(frames: List[np.ndarray], temp_folder: str,
                        prefix: str, start_num: int) -> int:
    for i, frame in enumerate(frames):
        path = get_frame_path(temp_folder, prefix, start_num + i)
        save_frame(frame, path)
    return start_num + len(frames)


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    if not frames:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    
    stacked = np.stack(frames, axis=0)
    tensor = torch.from_numpy(stacked).float() / 255.0
    return tensor


def tensor_to_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    if tensor is None:
        return []
    
    frames = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return [frames[i] for i in range(frames.shape[0])]


def create_mask_batch(height: int, width: int) -> torch.Tensor:
    masks = []
    
    for _ in range(8):
        masks.append(np.zeros((height, width), dtype=np.float32))
    
    for _ in range(16):
        masks.append(np.ones((height, width), dtype=np.float32))
    
    for _ in range(9):
        masks.append(np.zeros((height, width), dtype=np.float32))
    
    stacked = np.stack(masks, axis=0)
    return torch.from_numpy(stacked)


def read_all_temp_frames(temp_folder: str, prefix: str) -> List[np.ndarray]:
    pattern = os.path.join(temp_folder, f"{prefix}_*.png")
    files = sorted(glob.glob(pattern))
    
    frames = []
    for file_path in files:
        img = Image.open(file_path)
        frames.append(np.array(img))
    
    return frames


def cleanup_temp_folder(temp_folder: str) -> None:
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)


# =============================================================================
# Node 1: WanVaceAutoJoiner (INIT + PROCESS combined)
# =============================================================================

class WanVaceAutoJoiner:
    """
    Combined INIT/PROCESS node for use inside Easy Use For Loop.
    
    - index=1: INIT - Create temp folder, save Part A, output VACE batch 1
    - index>1: PROCESS - Save previous VACE + Part F, output next VACE batch
    
    Uses 1-based indexing internally (adds 1 to For Loop's 0-based index).
    
    NOTE: This node is NOT in the flow path. Connect For Loop Start's index
    output to this node's loop_index input. The Save node handles the flow
    barrier.
    """
    
    CATEGORY = "WAN VACE/Auto Joiner"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "status", "is_complete")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "loop_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "Index from For Loop Start (0-based, converted to 1-based internally)"
                }),
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Directory containing the video files"
                }),
                "file_prefix": ("STRING", {
                    "default": "clip",
                    "multiline": False,
                    "tooltip": "Prefix of the video files (e.g., 'clip' for clip_00001.mp4)"
                }),
                "first_suffix": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 99999,
                    "tooltip": "First sequence number of video files"
                }),
                "last_suffix": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 99999,
                    "tooltip": "Last sequence number of video files"
                }),
            }
        }
    
    def process(self, loop_index: int, directory: str, file_prefix: str,
                first_suffix: int, last_suffix: int
                ) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
        """
        Main processing - routes to INIT or PROCESS based on index.
        
        Args:
            loop_index: 0-based index from For Loop Start
            directory: Path to video files
            file_prefix: Common prefix for video files
            first_suffix: First video number
            last_suffix: Last video number
        """
        
        # Convert 0-based to 1-based index
        index = loop_index + 1
        
        # Validate inputs
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")
        
        if first_suffix > last_suffix:
            raise ValueError("first_suffix must be <= last_suffix")
        
        num_videos = last_suffix - first_suffix + 1
        num_transitions = num_videos - 1
        
        if num_transitions < 1:
            raise ValueError("Need at least 2 videos to join")
        
        print(f"[WAN VACE Auto Joiner] Step {index}/{num_transitions}")
        
        if index == 1:
            return self._do_init(directory, file_prefix, first_suffix, 
                                  last_suffix, num_transitions)
        else:
            return self._do_process(directory, file_prefix, index, num_transitions)
    
    def _do_init(self, directory: str, file_prefix: str,
                 first_suffix: int, last_suffix: int, num_transitions: int
                 ) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
        """
        INIT: Create temp folder, save Part A, output first VACE batch.
        """
        
        print(f"[WAN VACE Auto Joiner] INIT - {num_transitions + 1} videos, {num_transitions} transitions")
        
        # Clean up any existing temp folder
        existing_temp = find_temp_folder(directory)
        if existing_temp:
            print(f"[WAN VACE Auto Joiner] Cleaning up existing temp folder")
            cleanup_temp_folder(existing_temp)
        
        # Create new temp folder
        temp_folder = create_temp_folder(directory)
        print(f"[WAN VACE Auto Joiner] Created: {temp_folder}")
        
        # Get first video info
        first_video_path = get_video_path(directory, file_prefix, first_suffix)
        width, height, x_frames, fps = get_video_info(first_video_path)
        print(f"[WAN VACE Auto Joiner] Video 1: {x_frames} frames, {width}x{height}")
        
        if x_frames <= OVERLAP_FRAMES:
            raise ValueError(f"Video 1 has only {x_frames} frames, need more than {OVERLAP_FRAMES} frames")
        
        # Save Part A: first x-16 frames
        frames_a = read_video_frames(first_video_path, 0, x_frames - OVERLAP_FRAMES)
        frame_counter = save_frames_to_temp(frames_a, temp_folder, file_prefix, 1)
        print(f"[WAN VACE Auto Joiner] Saved Part A: {len(frames_a)} frames")
        
        # Read Parts B+C: last 16 frames from video 1
        frames_bc = read_video_frames(first_video_path, 
                                       x_frames - OVERLAP_FRAMES, x_frames)
        
        # Get video 2 info
        second_video_path = get_video_path(directory, file_prefix, first_suffix + 1)
        _, _, y_frames, _ = get_video_info(second_video_path)
        print(f"[WAN VACE Auto Joiner] Video 2: {y_frames} frames")
        
        if y_frames < NEXT_FRAMES:
            raise ValueError(f"Video 2 has only {y_frames} frames, need at least {NEXT_FRAMES} frames for VACE batch")
        
        # Read Parts D+E: first 17 frames from video 2
        frames_de = read_video_frames(second_video_path, 0, NEXT_FRAMES)
        
        # Build VACE batch: 16 + 17 = 33 frames
        image_list = frames_bc + frames_de
        
        # Replace frames [8:24] with gray
        gray_image = create_solid_image(width, height, (127, 127, 127))
        for i in range(MASK_START, MASK_END):
            image_list[i] = gray_image.copy()
        
        # Create tensors
        image_tensor = frames_to_tensor(image_list)
        mask_tensor = create_mask_batch(height, width)
        
        # Save state
        state = {
            "phase": "INIT_COMPLETE",
            "current_index": 1,
            "num_transitions": num_transitions,
            "frame_counter": frame_counter,
            "width": width,
            "height": height,
            "fps": fps,
            "first_suffix": first_suffix,
            "last_suffix": last_suffix,
            "current_video_frames": y_frames,  # Video 2's frame count
            "file_prefix": file_prefix
        }
        save_state(temp_folder, state)
        
        status = f"Step 1/{num_transitions}: INIT complete. VACE batch 1 ready."
        print(f"[WAN VACE Auto Joiner] {status}")
        
        return (image_tensor, mask_tensor, status, False)
    
    def _do_process(self, directory: str, file_prefix: str,
                    index: int, num_transitions: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
        """
        PROCESS: Save previous VACE output + Part F, output next VACE batch.
        """
        
        # Find temp folder and load state
        temp_folder = find_temp_folder(directory)
        if not temp_folder:
            raise ValueError("No temp folder found. INIT must run first (index=1).")
        
        state = load_state(temp_folder)
        if not state:
            raise ValueError("No state file found. INIT must run first.")
        
        # Read saved VACE output from previous iteration
        vace_output_path = os.path.join(temp_folder, "vace_output.pt")
        if not os.path.exists(vace_output_path):
            raise ValueError("No VACE output found. Save node must run after VACE.")
        
        vace_images = torch.load(vace_output_path)
        os.remove(vace_output_path)
        
        # Extract state
        frame_counter = state["frame_counter"]
        width = state["width"]
        height = state["height"]
        first_suffix = state["first_suffix"]
        current_video_frames = state["current_video_frames"]
        
        print(f"[WAN VACE Auto Joiner] PROCESS - Step {index}/{num_transitions}")
        
        # Save VACE frames from previous transition
        vace_frames = tensor_to_frames(vace_images)
        frame_counter = save_frames_to_temp(vace_frames, temp_folder, 
                                             file_prefix, frame_counter)
        print(f"[WAN VACE Auto Joiner] Saved VACE: {len(vace_frames)} frames")
        
        # Current video is at position (first_suffix + index - 1)
        # index=2 means we just processed transition 1→2, now doing 2→3
        current_video_idx = first_suffix + index - 1
        current_video_path = get_video_path(directory, file_prefix, current_video_idx)
        y_frames = current_video_frames
        
        # Save Part F: middle frames from current video (y-33 frames)
        middle_start = NEXT_FRAMES
        middle_end = y_frames - OVERLAP_FRAMES
        
        if middle_end > middle_start:
            frames_f = read_video_frames(current_video_path, middle_start, middle_end)
            frame_counter = save_frames_to_temp(frames_f, temp_folder,
                                                 file_prefix, frame_counter)
            print(f"[WAN VACE Auto Joiner] Saved Part F: {len(frames_f)} frames")
        
        # Get next video info
        next_video_idx = first_suffix + index
        next_video_path = get_video_path(directory, file_prefix, next_video_idx)
        _, _, next_video_frames, _ = get_video_info(next_video_path)
        print(f"[WAN VACE Auto Joiner] Video {next_video_idx - first_suffix + 1}: {next_video_frames} frames")
        
        if next_video_frames < NEXT_FRAMES:
            raise ValueError(f"Video {next_video_idx - first_suffix + 1} has only {next_video_frames} frames, need at least {NEXT_FRAMES} frames for VACE batch")
        
        # Build next VACE batch
        # Parts G+H: last 16 frames from current video
        frames_gh = read_video_frames(current_video_path,
                                       y_frames - OVERLAP_FRAMES,
                                       y_frames)
        # Parts I+J: first 17 frames from next video
        frames_ij = read_video_frames(next_video_path, 0, NEXT_FRAMES)
        
        image_list = frames_gh + frames_ij
        
        # Replace [8:24] with gray
        gray_image = create_solid_image(width, height, (127, 127, 127))
        for i in range(MASK_START, MASK_END):
            image_list[i] = gray_image.copy()
        
        # Create tensors
        image_tensor = frames_to_tensor(image_list)
        mask_tensor = create_mask_batch(height, width)
        
        # Update state
        state["phase"] = "PROCESSING"
        state["current_index"] = index
        state["frame_counter"] = frame_counter
        state["current_video_frames"] = next_video_frames
        save_state(temp_folder, state)
        
        status = f"Step {index}/{num_transitions}: PROCESS complete. VACE batch {index} ready."
        print(f"[WAN VACE Auto Joiner] {status}")
        
        return (image_tensor, mask_tensor, status, False)


# =============================================================================
# Node 2: WanVaceAutoJoinerSave (Inside Loop)
# =============================================================================

class WanVaceAutoJoinerSave:
    """
    Save node - saves VACE output to disk for next iteration or Finalize.
    
    This node runs INSIDE the For Loop, after VACE processing.
    It saves the VACE output to vace_output.pt for:
    - The next PROCESS iteration to read
    - Or the FINALIZE node to read after the loop
    
    *** THIS NODE IS THE BARRIER VIA value1 ***
    
    Because it depends on vace_images (from VAE Decode), it won't execute
    until VACE processing is complete. By having value1 pass through this node,
    the For Loop End waits for each iteration's work to finish before advancing.
    
    Connection pattern:
    - flow path: For Loop Start [flow] → For Loop End [flow] (DIRECT!)
    - value1 path: For Loop Start [value1] → Save [value1] → For Loop End [initial_value1]
    """
    
    CATEGORY = "WAN VACE/Auto Joiner"
    FUNCTION = "process"
    # Output value1 passthrough (as wildcard to match For Loop types)
    RETURN_TYPES = ("*", "STRING", "BOOLEAN")
    RETURN_NAMES = ("value1", "status", "is_complete")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "value1": ("*", {
                    "tooltip": "Connect to For Loop Start's value1 output"
                }),
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Directory containing the video files (same as Auto Joiner)"
                }),
                "vace_images": ("IMAGE", {
                    "tooltip": "Output from VAE Decode after VACE processing"
                }),
            }
        }
    
    def process(self, value1, directory: str, vace_images: torch.Tensor
                ) -> Tuple[Any, str, bool]:
        """Save VACE output to disk and pass through value1 unchanged."""
        
        print(f"[WAN VACE Auto Joiner Save] Saving VACE output...")
        
        # Find temp folder
        temp_folder = find_temp_folder(directory)
        if not temp_folder:
            raise ValueError("No temp folder found. Auto Joiner must run first.")
        
        # Save VACE output
        vace_output_path = os.path.join(temp_folder, "vace_output.pt")
        torch.save(vace_images, vace_output_path)
        
        num_frames = vace_images.shape[0]
        status = f"Saved {num_frames} VACE frames to disk."
        print(f"[WAN VACE Auto Joiner Save] {status}")
        
        # Pass through value1 unchanged - this creates the barrier!
        return (value1, status, False)


# =============================================================================
# Node 3: WanVaceAutoJoinerFinalize (After Loop)
# =============================================================================

class WanVaceAutoJoinerFinalize:
    """
    Finalize node - runs AFTER the For Loop with NO VACE overhead!
    
    - Reads the final VACE output from disk
    - Saves it to temp folder
    - Saves Part K (z-17 frames from last video)
    - Outputs ALL frames as batch_images
    
    Connect For Loop End's value1 output DIRECTLY to loop_end_trigger.
    This ensures Finalize only runs after all loop iterations complete.
    
    Prefers reading VACE output from disk (vace_output.pt).
    """
    
    CATEGORY = "WAN VACE/Auto Joiner"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("batch_images", "status", "frame_rate", "is_complete")
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "loop_end_trigger": ("*", {
                    "tooltip": "Connect DIRECTLY to For Loop End's value1 output"
                }),
                "directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Directory containing the video files (same as Auto Joiner)"
                }),
                "file_prefix": ("STRING", {
                    "default": "clip",
                    "multiline": False,
                    "tooltip": "Prefix of the video files (same as Auto Joiner)"
                }),
                "cleanup": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Delete temp folder after outputting final video"
                }),
            },
            "optional": {
                "vace_images": ("IMAGE", {
                    "tooltip": "Optional: Final VACE output (if not provided, reads from disk)"
                }),
            }
        }
    
    def process(self, loop_end_trigger, directory: str, file_prefix: str, cleanup: bool,
                vace_images: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, str, float, bool]:
        """
        Finalize: Save final VACE output + Part K, output all frames.
        """
        
        print(f"[WAN VACE Auto Joiner Finalize] Starting finalization...")
        
        # Find temp folder
        temp_folder = find_temp_folder(directory)
        if not temp_folder:
            raise ValueError("No temp folder found. Processing must complete first.")
        
        # Load state
        state = load_state(temp_folder)
        if not state:
            raise ValueError("No state file found.")
        
        frame_counter = state["frame_counter"]
        fps = state.get("fps", 16.0)
        first_suffix = state["first_suffix"]
        last_suffix = state["last_suffix"]
        current_video_frames = state["current_video_frames"]
        num_transitions = state["num_transitions"]
        
        print(f"[WAN VACE Auto Joiner Finalize] Step {num_transitions + 1}/{num_transitions + 1} (FINALIZE)")
        
        # Get final VACE output - prefer disk, fallback to input
        vace_output_path = os.path.join(temp_folder, "vace_output.pt")
        
        if os.path.exists(vace_output_path):
            print(f"[WAN VACE Auto Joiner Finalize] Reading VACE output from disk")
            final_vace = torch.load(vace_output_path)
            os.remove(vace_output_path)
        elif vace_images is not None and vace_images.shape[0] >= TOTAL_BATCH:
            print(f"[WAN VACE Auto Joiner Finalize] Using VACE output from input")
            final_vace = vace_images
        else:
            raise ValueError("No VACE output found. Save node must run after last VACE.")
        
        # Save final VACE frames
        vace_frames = tensor_to_frames(final_vace)
        frame_counter = save_frames_to_temp(vace_frames, temp_folder, 
                                             file_prefix, frame_counter)
        print(f"[WAN VACE Auto Joiner Finalize] Saved final VACE: {len(vace_frames)} frames")
        
        # Save Part K: remaining frames from last video (z-17)
        last_video_idx = last_suffix
        last_video_path = get_video_path(directory, file_prefix, last_video_idx)
        z_frames = current_video_frames  # Last video's frame count
        
        if z_frames > NEXT_FRAMES:
            frames_k = read_video_frames(last_video_path, NEXT_FRAMES, z_frames)
            frame_counter = save_frames_to_temp(frames_k, temp_folder,
                                                 file_prefix, frame_counter)
            print(f"[WAN VACE Auto Joiner Finalize] Saved Part K: {len(frames_k)} frames")
        
        # Read ALL frames from temp folder
        print(f"[WAN VACE Auto Joiner Finalize] Reading all frames from temp folder...")
        all_frames = read_all_temp_frames(temp_folder, file_prefix)
        
        if not all_frames:
            raise ValueError("No frames found in temp folder.")
        
        batch_tensor = frames_to_tensor(all_frames)
        print(f"[WAN VACE Auto Joiner Finalize] Total output: {len(all_frames)} frames at {fps} fps")
        
        # Mark as finalized
        state["phase"] = "FINALIZED"
        save_state(temp_folder, state)
        
        # Cleanup if requested
        if cleanup:
            cleanup_temp_folder(temp_folder)
            print(f"[WAN VACE Auto Joiner Finalize] Cleaned up temp folder")
            status = f"DONE! Output {len(all_frames)} frames. Temp folder deleted."
        else:
            status = f"DONE! Output {len(all_frames)} frames at {fps} fps."
        
        return (batch_tensor, status, fps, True)
