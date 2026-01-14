"""
WAN VACE Auto Joiner - ComfyUI Custom Nodes

Seamlessly join multiple video clips using WAN VACE with one click.

v2.0.0 - Major Release:
- NEW: Temporal color smoothing for seamless transitions
- NEW: Per-channel (R, G, B) color correction
- NEW: Audio transfer from original clips
- NEW: Standard ComfyUI AUDIO output for Video Combine
- NEW: Fail-safe audio (silent track when no source audio)
- Dynamic correction values from source frames
- Requires ffmpeg for audio features
- Input sanitization for security
"""

from .wan_vace_auto_joiner import (
    WanVaceAutoJoiner,
    WanVaceAutoJoinerSave,
    WanVaceAutoJoinerFinalize
)

NODE_CLASS_MAPPINGS = {
    "WanVaceAutoJoiner": WanVaceAutoJoiner,
    "WanVaceAutoJoinerSave": WanVaceAutoJoinerSave,
    "WanVaceAutoJoinerFinalize": WanVaceAutoJoinerFinalize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVaceAutoJoiner": "WAN VACE Auto Joiner",
    "WanVaceAutoJoinerSave": "WAN VACE Auto Joiner - Save",
    "WanVaceAutoJoinerFinalize": "WAN VACE Auto Joiner - Finalize"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
