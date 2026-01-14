"""
WAN VACE Auto Joiner - ComfyUI Custom Nodes

Seamlessly join multiple video clips using WAN VACE with one click.

Nodes:
- WanVaceAutoJoiner: Prepares frames for VACE processing (inside loop)
- WanVaceAutoJoinerSave: Saves VACE output between iterations (inside loop)
- WanVaceAutoJoinerFinalize: Outputs final joined video (after loop)

For N videos, VACE runs exactly N-1 times (one per transition).

v2.0.0 - Major Release:
- NEW: Temporal color smoothing for seamless transitions
- NEW: Per-channel (R, G, B) correction with Gaussian + linear interpolation
- NEW: Audio transfer from original clips to final video
- NEW: Standard ComfyUI AUDIO output for Video Combine compatibility
- NEW: Fail-safe audio handling (silent track when source has no audio)
- All correction values calculated dynamically from source frames
- Requires ffmpeg for audio features
- Input sanitization for security
"""

import os
import json
import glob
import shutil
import re
import subprocess
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import torch
import numpy as np
from PIL import Image
import cv2

# Try to import scipy for Gaussian smoothing, fallback to numpy if not available
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WAN VACE Auto Joiner] scipy not found, using numpy fallback for smoothing")

# Try to import ComfyUI folder paths for security validation
try:
    import folder_paths
    COMFYUI_INPUT_DIR = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    COMFYUI_INPUT_DIR = None
    COMFYUI_OUTPUT_DIR = None


# =============================================================================
# Shared Constants
# =============================================================================

OVERLAP_FRAMES = 16
NEXT_FRAMES = 17
TOTAL_BATCH = 33
MASK_START = 8
MASK_END = 24


# =============================================================================
# Security Helper Functions
# =============================================================================

def sanitize_prefix(prefix: str) -> str:
    """
    Sanitize file prefix to prevent path traversal attacks.
    Removes any path separators and dangerous characters.
    """
    prefix = prefix.replace("/", "").replace("\\", "")
    prefix = prefix.replace("..", "")
    prefix = re.sub(r'[^a-zA-Z0-9_\-]', '', prefix)
    
    if not prefix:
        raise ValueError("Invalid file prefix: prefix cannot be empty after sanitization")
    
    return prefix


def validate_directory(directory: str) -> str:
    """
    Validate and normalize the directory path.
    Returns the absolute, resolved path.
    """
    if not directory or not directory.strip():
        raise ValueError("Directory path cannot be empty")
    
    abs_path = os.path.realpath(os.path.abspath(directory))
    
    if not os.path.isdir(abs_path):
        raise ValueError(f"Directory does not exist: {directory}")
    
    return abs_path


def validate_path_within_directory(path: str, base_directory: str) -> str:
    """
    Ensure the given path is within the base directory.
    Prevents path traversal attacks.
    """
    abs_path = os.path.realpath(os.path.abspath(path))
    abs_base = os.path.realpath(os.path.abspath(base_directory))
    
    if not abs_path.startswith(abs_base + os.sep) and abs_path != abs_base:
        raise ValueError(f"Path traversal detected: {path} is outside {base_directory}")
    
    return abs_path


# =============================================================================
# Shared Helper Functions
# =============================================================================

def get_video_path(directory: str, prefix: str, suffix: int) -> str:
    """Get video path with security validation."""
    filename = f"{prefix}_{suffix:05d}.mp4"
    full_path = os.path.join(directory, filename)
    validate_path_within_directory(full_path, directory)
    return full_path


def get_frame_path(temp_folder: str, prefix: str, frame_num: int) -> str:
    """Get frame path with security validation."""
    filename = f"{prefix}_{frame_num:05d}.png"
    full_path = os.path.join(temp_folder, filename)
    validate_path_within_directory(full_path, temp_folder)
    return full_path


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
    """Get video dimensions, frame count, and fps."""
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
    """Save frames to temp folder with sequential numbering."""
    frame_num = start_num
    for frame in frames:
        frame_path = get_frame_path(temp_folder, prefix, frame_num)
        save_frame(frame, frame_path)
        frame_num += 1
    return frame_num


def read_all_temp_frames(temp_folder: str, prefix: str) -> List[np.ndarray]:
    """Read all frames from temp folder in order."""
    pattern = os.path.join(temp_folder, f"{prefix}_*.png")
    frame_files = sorted(glob.glob(pattern))
    
    frames = []
    for file_path in frame_files:
        img = Image.open(file_path)
        frame = np.array(img)
        frames.append(frame)
    
    return frames


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert list of numpy frames to ComfyUI tensor format."""
    if not frames:
        raise ValueError("No frames to convert")
    
    frames_np = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    return frames_tensor


def tensor_to_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    """Convert ComfyUI tensor to list of numpy frames."""
    frames_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return [frames_np[i] for i in range(frames_np.shape[0])]


def create_mask_batch(height: int, width: int) -> torch.Tensor:
    """Create VACE mask batch: 33 frames with [8:24] masked."""
    masks = []
    for i in range(TOTAL_BATCH):
        if MASK_START <= i < MASK_END:
            mask = np.ones((height, width), dtype=np.float32)
        else:
            mask = np.zeros((height, width), dtype=np.float32)
        masks.append(mask)
    
    masks_np = np.stack(masks, axis=0)
    return torch.from_numpy(masks_np)


def cleanup_temp_folder(temp_folder: str) -> None:
    """Remove temp folder and all contents."""
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)


# =============================================================================
# Temporal Color Smoothing Algorithm
# =============================================================================

def numpy_gaussian_filter1d(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Simple Gaussian filter implementation using numpy.
    Fallback when scipy is not available.
    """
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    # Pad data
    pad_size = kernel_size // 2
    padded = np.pad(data, pad_size, mode='edge')
    
    # Convolve
    result = np.convolve(padded, kernel, mode='valid')
    
    return result


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing, using scipy if available."""
    if HAS_SCIPY:
        return gaussian_filter1d(data, sigma=sigma)
    else:
        return numpy_gaussian_filter1d(data, sigma)


def temporal_smooth_color(frames: List[np.ndarray], 
                          transition_start: int, 
                          transition_end: int,
                          smooth_window: int = 12,
                          blend_region: int = 25) -> List[np.ndarray]:
    """
    Apply temporal smoothing to R, G, B channels independently.
    
    This algorithm:
    1. Calculates current R, G, B values for each frame in the region
    2. Creates smooth target curves using Gaussian smoothing + linear interpolation
    3. Applies per-frame, per-channel correction factors
    
    All values are calculated dynamically from the actual frame data.
    No hardcoded adjustment values.
    
    Args:
        frames: List of all video frames (numpy arrays, RGB format)
        transition_start: First frame of VACE region
        transition_end: First frame after VACE region
        smooth_window: Gaussian smoothing sigma (higher = smoother)
        blend_region: Number of frames before/after VACE to include in smoothing
    
    Returns:
        List of corrected frames (same length as input)
    """
    # Make a copy to avoid modifying original
    result_frames = [f.copy() for f in frames]
    
    # Define the region to process (VACE region + context)
    region_start = max(0, transition_start - blend_region)
    region_end = min(len(frames), transition_end + blend_region)
    
    # Calculate current values for each channel
    r_vals, g_vals, b_vals = [], [], []
    for i in range(region_start, region_end):
        frame = frames[i].astype(np.float32)
        r_vals.append(np.mean(frame[:,:,0]))
        g_vals.append(np.mean(frame[:,:,1]))
        b_vals.append(np.mean(frame[:,:,2]))
    
    r_vals = np.array(r_vals)
    g_vals = np.array(g_vals)
    b_vals = np.array(b_vals)
    
    # Get VACE region indices relative to our processing region
    vace_start_idx = transition_start - region_start
    vace_end_idx = transition_end - region_start
    
    def create_target_curve(vals: np.ndarray, 
                           vace_start: int, 
                           vace_end: int, 
                           sigma: float) -> np.ndarray:
        """
        Create a smooth target curve for one channel.
        
        Combines:
        1. Gaussian smoothing of the original values
        2. Linear interpolation through the VACE region
        """
        # Apply Gaussian smoothing
        smoothed = gaussian_smooth(vals, sigma=sigma)
        
        # Calculate anchor points from frames outside VACE region
        # Use average of several frames for stability
        anchor_frames = min(8, vace_start)  # Use up to 8 frames
        before_val = np.mean(vals[:vace_start]) if vace_start > 0 else vals[0]
        after_val = np.mean(vals[vace_end:]) if vace_end < len(vals) else vals[-1]
        
        # Create linear interpolation through VACE region
        linear = np.copy(smoothed)
        for i in range(vace_start, vace_end):
            progress = (i - vace_start) / max(1, vace_end - vace_start)
            linear[i] = before_val * (1 - progress) + after_val * progress
        
        # Blend smoothed and linear approaches (50/50)
        # This preserves some natural variation while ensuring smooth transitions
        target = 0.5 * smoothed + 0.5 * linear
        
        return target
    
    # Create target curves for each channel
    r_target = create_target_curve(r_vals, vace_start_idx, vace_end_idx, smooth_window)
    g_target = create_target_curve(g_vals, vace_start_idx, vace_end_idx, smooth_window)
    b_target = create_target_curve(b_vals, vace_start_idx, vace_end_idx, smooth_window)
    
    # Apply corrections to each frame in the region
    for i in range(region_start, region_end):
        idx = i - region_start
        frame = frames[i].astype(np.float32)
        
        # Calculate per-channel correction factors
        # These are computed dynamically from the actual frame data
        r_factor = r_target[idx] / r_vals[idx] if r_vals[idx] > 0 else 1.0
        g_factor = g_target[idx] / g_vals[idx] if g_vals[idx] > 0 else 1.0
        b_factor = b_target[idx] / b_vals[idx] if b_vals[idx] > 0 else 1.0
        
        # Apply correction
        corrected = frame.copy()
        corrected[:,:,0] = np.clip(frame[:,:,0] * r_factor, 0, 255)
        corrected[:,:,1] = np.clip(frame[:,:,1] * g_factor, 0, 255)
        corrected[:,:,2] = np.clip(frame[:,:,2] * b_factor, 0, 255)
        
        result_frames[i] = corrected.astype(np.uint8)
    
    return result_frames


def apply_all_transition_smoothing(frames: List[np.ndarray],
                                   vace_regions: List[Dict[str, int]],
                                   smooth_window: int = 12,
                                   blend_region: int = 25) -> List[np.ndarray]:
    """
    Apply temporal color smoothing to all VACE transition regions.
    
    Args:
        frames: List of all video frames
        vace_regions: List of dicts with 'start' and 'end' keys for each VACE region
        smooth_window: Gaussian smoothing sigma
        blend_region: Context frames to include
    
    Returns:
        List of corrected frames
    """
    result = frames
    
    for region in vace_regions:
        result = temporal_smooth_color(
            result,
            region['start'],
            region['end'],
            smooth_window=smooth_window,
            blend_region=blend_region
        )
    
    return result


# =============================================================================
# Audio Transfer Functions
# =============================================================================

def extract_audio_from_videos(video_paths: List[str], output_audio_path: str) -> bool:
    """
    Extract and concatenate audio from multiple videos using ffmpeg.
    
    Args:
        video_paths: List of input video file paths
        output_audio_path: Path for combined audio output (WAV format for tensor loading)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WAN VACE Auto Joiner] ffmpeg not found, skipping audio transfer")
        return False
    
    if not video_paths:
        return False
    
    temp_audio_files = []
    
    try:
        for i, video_path in enumerate(video_paths):
            if not os.path.exists(video_path):
                continue
            
            # Extract to WAV format for easy tensor loading
            temp_audio = output_audio_path.replace('.wav', f'_temp_{i}.wav')
            temp_audio_files.append(temp_audio)
            
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                temp_audio
            ]
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode != 0:
                temp_audio_files.pop()
        
        if not temp_audio_files:
            return False
        
        if len(temp_audio_files) == 1:
            shutil.copy(temp_audio_files[0], output_audio_path)
        else:
            concat_list_path = output_audio_path.replace('.wav', '_list.txt')
            with open(concat_list_path, 'w') as f:
                for audio_file in temp_audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list_path,
                '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                output_audio_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            os.remove(concat_list_path)
        
        return True
        
    except Exception as e:
        print(f"[WAN VACE Auto Joiner] Audio extraction error: {e}")
        return False
        
    finally:
        for temp_file in temp_audio_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def load_audio_as_tensor(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Load audio file and convert to ComfyUI AUDIO format.
    
    ComfyUI AUDIO format: {"waveform": torch.Tensor[B,C,T], "sample_rate": int}
    
    Args:
        audio_path: Path to WAV audio file
    
    Returns:
        Dictionary with waveform tensor and sample_rate, or None if failed
    """
    try:
        import scipy.io.wavfile as wavfile
        
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        else:
            audio_data = audio_data.astype(np.float32)
        
        # Handle mono vs stereo
        if len(audio_data.shape) == 1:
            # Mono: shape (T,) -> (1, 1, T)
            waveform = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
        else:
            # Stereo: shape (T, C) -> (1, C, T)
            waveform = torch.from_numpy(audio_data.T).unsqueeze(0)
        
        return {"waveform": waveform, "sample_rate": sample_rate}
        
    except ImportError:
        # Fallback without scipy - use wave module
        try:
            import wave
            import struct
            
            with wave.open(audio_path, 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                raw_data = wav_file.readframes(n_frames)
            
            # Parse based on sample width
            if sample_width == 2:
                fmt = f'<{n_frames * n_channels}h'
                audio_data = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
            elif sample_width == 1:
                audio_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
                audio_data = (audio_data - 128) / 128.0
            else:
                print(f"[WAN VACE Auto Joiner] Unsupported sample width: {sample_width}")
                return None
            
            # Reshape for channels
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels).T  # Shape: (C, T)
                waveform = torch.from_numpy(audio_data).unsqueeze(0)  # Shape: (1, C, T)
            else:
                waveform = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T)
            
            return {"waveform": waveform, "sample_rate": sample_rate}
            
        except Exception as e:
            print(f"[WAN VACE Auto Joiner] Failed to load audio: {e}")
            return None
    
    except Exception as e:
        print(f"[WAN VACE Auto Joiner] Failed to load audio: {e}")
        return None


def create_silent_audio(duration_seconds: float, sample_rate: int = 44100, channels: int = 2) -> Dict[str, Any]:
    """
    Create a silent audio track of specified duration.
    
    Args:
        duration_seconds: Length of silent audio in seconds
        sample_rate: Audio sample rate (default 44100 Hz)
        channels: Number of audio channels (default 2 for stereo)
    
    Returns:
        Dictionary with silent waveform tensor and sample_rate in ComfyUI AUDIO format
    """
    # Calculate number of samples
    num_samples = int(duration_seconds * sample_rate)
    
    # Create silent waveform (zeros)
    # Shape: [batch=1, channels, samples]
    waveform = torch.zeros(1, channels, num_samples, dtype=torch.float32)
    
    print(f"[WAN VACE Auto Joiner] Created silent audio: {duration_seconds:.2f}s, {sample_rate}Hz, {channels}ch")
    
    return {"waveform": waveform, "sample_rate": sample_rate}


# =============================================================================
# Node 1: WanVaceAutoJoiner (INIT + PROCESS combined)
# =============================================================================

class WanVaceAutoJoiner:
    """
    Main processing node - prepares frames for WAN VACE.
    
    Place this node inside the For Loop.
    
    Connections:
    - loop_index: Connect to For Loop Start's index output
    - image output: Connect to WanVaceToVideo control_video input
    - mask output: Connect to WanVaceToVideo control_mask input
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
                    "tooltip": "Index from For Loop Start (0-based)"
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
        """Main processing - routes to INIT or PROCESS based on index."""
        
        # Sanitize and validate inputs
        directory = validate_directory(directory)
        file_prefix = sanitize_prefix(file_prefix)
        
        # Convert 0-based to 1-based index
        index = loop_index + 1
        
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
        """INIT: Create temp folder, save Part A, output first VACE batch."""
        
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
        
        # Track VACE region position (for temporal smoothing later)
        # First VACE region starts at frame_counter (1-indexed, so subtract 1 for 0-indexed)
        vace_region_start = frame_counter - 1  # 0-indexed position in final assembly
        
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
        
        # Save state with VACE region tracking
        state = {
            "phase": "INIT",
            "directory": directory,
            "file_prefix": file_prefix,
            "first_suffix": first_suffix,
            "last_suffix": last_suffix,
            "num_transitions": num_transitions,
            "frame_counter": frame_counter,
            "width": width,
            "height": height,
            "fps": fps,
            "current_video_frames": y_frames,
            "vace_regions": [{"start": vace_region_start, "end": vace_region_start + TOTAL_BATCH}],
        }
        save_state(temp_folder, state)
        
        status = f"Step 1/{num_transitions}: INIT complete. VACE batch 1 ready."
        print(f"[WAN VACE Auto Joiner] {status}")
        
        return (image_tensor, mask_tensor, status, False)
    
    def _do_process(self, directory: str, file_prefix: str,
                    index: int, num_transitions: int
                    ) -> Tuple[torch.Tensor, torch.Tensor, str, bool]:
        """PROCESS: Save previous VACE output + Part F, output next VACE batch."""
        
        # Find temp folder
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
        vace_regions = state.get("vace_regions", [])
        
        print(f"[WAN VACE Auto Joiner] PROCESS - Step {index}/{num_transitions}")
        
        # Save VACE frames from previous transition
        vace_frames = tensor_to_frames(vace_images)
        frame_counter = save_frames_to_temp(vace_frames, temp_folder, 
                                             file_prefix, frame_counter)
        print(f"[WAN VACE Auto Joiner] Saved VACE: {len(vace_frames)} frames")
        
        # Current video is at position (first_suffix + index - 1)
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
        
        # Track next VACE region position
        vace_region_start = frame_counter - 1  # 0-indexed position
        
        # Get next video info
        next_video_idx = first_suffix + index
        next_video_path = get_video_path(directory, file_prefix, next_video_idx)
        _, _, next_video_frames, _ = get_video_info(next_video_path)
        print(f"[WAN VACE Auto Joiner] Video {next_video_idx - first_suffix + 1}: {next_video_frames} frames")
        
        if next_video_frames < NEXT_FRAMES:
            raise ValueError(f"Video {next_video_idx - first_suffix + 1} has only {next_video_frames} frames, need at least {NEXT_FRAMES} frames for VACE batch")
        
        # Build next VACE batch
        frames_gh = read_video_frames(current_video_path,
                                       y_frames - OVERLAP_FRAMES,
                                       y_frames)
        frames_ij = read_video_frames(next_video_path, 0, NEXT_FRAMES)
        
        image_list = frames_gh + frames_ij
        
        # Replace [8:24] with gray
        gray_image = create_solid_image(width, height, (127, 127, 127))
        for i in range(MASK_START, MASK_END):
            image_list[i] = gray_image.copy()
        
        # Create tensors
        image_tensor = frames_to_tensor(image_list)
        mask_tensor = create_mask_batch(height, width)
        
        # Update state with new VACE region
        vace_regions.append({"start": vace_region_start, "end": vace_region_start + TOTAL_BATCH})
        
        state["phase"] = "PROCESSING"
        state["current_index"] = index
        state["frame_counter"] = frame_counter
        state["current_video_frames"] = next_video_frames
        state["vace_regions"] = vace_regions
        save_state(temp_folder, state)
        
        status = f"Step {index}/{num_transitions}: PROCESS complete. VACE batch {index} ready."
        print(f"[WAN VACE Auto Joiner] {status}")
        
        return (image_tensor, mask_tensor, status, False)


# =============================================================================
# Node 2: WanVaceAutoJoinerSave (Inside Loop)
# =============================================================================

class WanVaceAutoJoinerSave:
    """
    Saves VACE output to disk between loop iterations.
    
    Place this node inside the For Loop, after VAE Decode.
    
    Connections:
    - value1: Connect to For Loop Start's value1 output
    - vace_images: Connect to VAE Decode output
    - value1 output: Connect to For Loop End's initial_value1 input
    """
    
    CATEGORY = "WAN VACE/Auto Joiner"
    FUNCTION = "process"
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
        """Save VACE output to disk."""
        
        # Validate directory
        directory = validate_directory(directory)
        
        print(f"[WAN VACE Auto Joiner Save] Saving VACE output...")
        
        # Find temp folder
        temp_folder = find_temp_folder(directory)
        if not temp_folder:
            raise ValueError("No temp folder found. Auto Joiner must run first.")
        
        # Validate temp folder is within directory
        validate_path_within_directory(temp_folder, directory)
        
        # Save VACE output
        vace_output_path = os.path.join(temp_folder, "vace_output.pt")
        validate_path_within_directory(vace_output_path, temp_folder)
        torch.save(vace_images, vace_output_path)
        
        num_frames = vace_images.shape[0]
        status = f"Saved {num_frames} VACE frames to disk."
        print(f"[WAN VACE Auto Joiner Save] {status}")
        
        # Pass through value1 unchanged
        return (value1, status, False)


# =============================================================================
# Node 3: WanVaceAutoJoinerFinalize (After Loop)
# =============================================================================

class WanVaceAutoJoinerFinalize:
    """
    Outputs the final joined video after loop completion.
    Applies temporal color smoothing for seamless transitions.
    Optionally transfers audio from original clips.
    
    Place this node AFTER the For Loop (not inside it).
    No additional VACE processing required.
    
    Connections:
    - loop_end_trigger: Connect to For Loop End's value1 output
    - batch_images output: Connect to VHS Video Combine images input
    - audio output: Connect to VHS Video Combine audio input
    - frame_rate output: Connect to VHS Video Combine frame_rate input
    """
    
    CATEGORY = "WAN VACE/Auto Joiner"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("batch_images", "audio", "frame_rate", "status", "is_complete")
    
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
                "smooth_transitions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply temporal color smoothing for seamless transitions"
                }),
                "smooth_window": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 30,
                    "tooltip": "Smoothing strength (higher = smoother, default 12)"
                }),
                "blend_region": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 50,
                    "tooltip": "Frames before/after VACE to include in smoothing (default 25)"
                }),
                "transfer_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Transfer audio from original clips (requires ffmpeg)"
                }),
            }
        }
    
    def process(self, loop_end_trigger, directory: str, file_prefix: str, cleanup: bool,
                vace_images: Optional[torch.Tensor] = None,
                smooth_transitions: bool = True,
                smooth_window: int = 12,
                blend_region: int = 25,
                transfer_audio: bool = True
                ) -> Tuple[torch.Tensor, Any, float, str, bool]:
        """Finalize: Save final VACE output + Part K, apply smoothing, output all frames."""
        
        # Sanitize and validate inputs
        directory = validate_directory(directory)
        file_prefix = sanitize_prefix(file_prefix)
        
        print(f"[WAN VACE Auto Joiner Finalize] Starting finalization...")
        
        # Find temp folder
        temp_folder = find_temp_folder(directory)
        if not temp_folder:
            raise ValueError("No temp folder found. Processing must complete first.")
        
        # Validate temp folder is within directory
        validate_path_within_directory(temp_folder, directory)
        
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
        vace_regions = state.get("vace_regions", [])
        
        print(f"[WAN VACE Auto Joiner Finalize] Step {num_transitions + 1}/{num_transitions + 1} (FINALIZE)")
        
        # Get final VACE output - prefer disk, fallback to input
        vace_output_path = os.path.join(temp_folder, "vace_output.pt")
        validate_path_within_directory(vace_output_path, temp_folder)
        
        if os.path.exists(vace_output_path):
            print(f"[WAN VACE Auto Joiner Finalize] Reading VACE output from disk")
            final_vace = torch.load(vace_output_path)
            os.remove(vace_output_path)
        elif vace_images is not None and vace_images.shape[0] >= TOTAL_BATCH:
            print(f"[WAN VACE Auto Joiner Finalize] Using VACE output from input")
            final_vace = vace_images
        else:
            raise ValueError("No VACE output found. Save node must run after last VACE.")
        
        # Track final VACE region position
        final_vace_start = frame_counter - 1
        vace_regions.append({"start": final_vace_start, "end": final_vace_start + TOTAL_BATCH})
        
        # Save final VACE frames
        vace_frames = tensor_to_frames(final_vace)
        frame_counter = save_frames_to_temp(vace_frames, temp_folder, 
                                             file_prefix, frame_counter)
        print(f"[WAN VACE Auto Joiner Finalize] Saved final VACE: {len(vace_frames)} frames")
        
        # Save Part K: remaining frames from last video (z-17)
        last_video_idx = last_suffix
        last_video_path = get_video_path(directory, file_prefix, last_video_idx)
        z_frames = current_video_frames
        
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
        
        print(f"[WAN VACE Auto Joiner Finalize] Loaded {len(all_frames)} frames")
        
        # Apply temporal color smoothing for seamless transitions
        if smooth_transitions and vace_regions:
            print(f"[WAN VACE Auto Joiner Finalize] Applying temporal color smoothing...")
            print(f"[WAN VACE Auto Joiner Finalize] VACE regions: {vace_regions}")
            print(f"[WAN VACE Auto Joiner Finalize] Smooth window: {smooth_window}, Blend region: {blend_region}")
            
            all_frames = apply_all_transition_smoothing(
                all_frames,
                vace_regions,
                smooth_window=smooth_window,
                blend_region=blend_region
            )
            print(f"[WAN VACE Auto Joiner Finalize] Smoothing complete")
        
        batch_tensor = frames_to_tensor(all_frames)
        print(f"[WAN VACE Auto Joiner Finalize] Total output: {len(all_frames)} frames at {fps} fps")
        
        # Transfer audio from original clips
        audio_output = None  # Will be a dict {"waveform": tensor, "sample_rate": int}
        audio_status = ""
        
        # Calculate video duration for silent audio fallback
        video_duration = len(all_frames) / fps if fps > 0 else 0
        
        if transfer_audio:
            print(f"[WAN VACE Auto Joiner Finalize] Extracting audio from original clips...")
            video_paths = [
                get_video_path(directory, file_prefix, i)
                for i in range(first_suffix, last_suffix + 1)
            ]
            # Use WAV format for easy tensor loading
            audio_output_path = os.path.join(temp_folder, "combined_audio.wav")
            
            if extract_audio_from_videos(video_paths, audio_output_path):
                # Load audio as tensor
                audio_output = load_audio_as_tensor(audio_output_path)
                
                if audio_output is not None:
                    state["audio_path"] = audio_output_path
                    save_state(temp_folder, state)
                    audio_status = " Audio extracted."
                    print(f"[WAN VACE Auto Joiner Finalize] Audio loaded: {audio_output['waveform'].shape}, {audio_output['sample_rate']} Hz")
                else:
                    # Audio file exists but failed to load - use silent fallback
                    print(f"[WAN VACE Auto Joiner Finalize] Audio load failed, using silent track")
                    audio_output = create_silent_audio(video_duration)
                    audio_status = " (Silent - audio load failed)"
            else:
                # No audio could be extracted - use silent fallback
                print(f"[WAN VACE Auto Joiner Finalize] No audio in source clips, using silent track")
                audio_output = create_silent_audio(video_duration)
                audio_status = " (Silent - no source audio)"
        else:
            # Audio transfer disabled - still provide silent track for compatibility
            print(f"[WAN VACE Auto Joiner Finalize] Audio transfer disabled, using silent track")
            audio_output = create_silent_audio(video_duration)
            audio_status = " (Silent)"
        
        # Mark as finalized
        state["phase"] = "FINALIZED"
        state["vace_regions"] = vace_regions
        save_state(temp_folder, state)
        
        # Cleanup if requested
        if cleanup:
            cleanup_temp_folder(temp_folder)
            print(f"[WAN VACE Auto Joiner Finalize] Cleaned up temp folder")
            status = f"DONE! Output {len(all_frames)} frames. Temp folder deleted."
        else:
            smoothing_note = " (smoothed)" if smooth_transitions else ""
            status = f"DONE! Output {len(all_frames)} frames{smoothing_note} at {fps} fps.{audio_status}"
        
        print(f"[WAN VACE Auto Joiner Finalize] {status}")
        
        return (batch_tensor, audio_output, fps, status, True)
