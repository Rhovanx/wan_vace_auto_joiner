# Example Workflows

## example_workflow.json

A complete working workflow that demonstrates the WAN VACE Auto Joiner setup.

### How to Use

1. Download `example_workflow.json`
2. In ComfyUI, click **Load** and select the file
3. Adjust the `directory` path to point to your video clips folder
4. Set `first_suffix` and `last_suffix` to match your video numbering
5. Adjust `For Loop Start → total` to match your number of transitions (N-1)
6. Click **Queue**

### Prerequisites

Make sure you have:
- ComfyUI-Easy-Use installed (for For Loop nodes)
- WAN VACE model loaded
- Video clips named like `clip_00001.mp4`, `clip_00002.mp4`, etc.
