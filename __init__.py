"""
WAN VACE Auto Joiner - ComfyUI Custom Nodes

Seamlessly join multiple video clips using WAN VACE with one click.
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
