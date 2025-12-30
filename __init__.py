"""
WAN VACE Auto Joiner - For Loop Compatible System

Three nodes designed to work with ComfyUI-Easy-Use For Loop:

1. WanVaceAutoJoiner - INIT (index=1) or PROCESS (index>1) - inside loop
2. WanVaceAutoJoinerSave - Save VACE output to disk - inside loop
3. WanVaceAutoJoinerFinalize - Output final video - AFTER loop (NO VACE!)

For N videos (N-1 transitions):
- Set For Loop Start total = N-1
- Loop runs N-1 times with VACE processing
- Finalize runs ONCE after loop - ZERO VACE overhead!

Example for 3 videos:
- For Loop total = 2
- Loop iteration 1: INIT → VACE → Save
- Loop iteration 2: PROCESS → VACE → Save
- After loop: FINALIZE → Video Combine (NO VACE!)
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
