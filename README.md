# Wan Vace Auto Joiner (ComfyUI Custom Nodes)

Seamlessly join multiple video clips in a folder using **WAN VACE** and **ComfyUI Easy-Use For Loop** automation — with **one-click solution**.

This node set ensures:
- VACE runs **exactly N-1 times** for **N input videos**
- Finalization runs **once**, after the loop completes
- No premature loop exits
- No redundant VACE calls

Nodes appear under: **WAN VACE / Auto Joiner**

---

## Screenshots

> Add screenshots after setup (recommended).

```
assets/screenshots/workflow.png
assets/screenshots/output.png
```

---

## Key Features

- One-click batch video joining
- Designed specifically for **WAN VACE** workflows
- Built-in **loop barrier** using `value1` (not FLOW_CONTROL)
- Prevents early loop exit & race conditions
- Clean INIT → PROCESS → FINALIZE lifecycle
- Safe temporary directory handling

---

## Requirements

- **ComfyUI**
- **ComfyUI-Manager** (recommended)
- **ComfyUI-Easy-Use** (required for For Loop)
- A working WAN VACE workflow

---

## Installation

### Option 1 — Install via ComfyUI-Manager (recommended)

1. Open **ComfyUI → Manager**
2. Go to **Install Custom Nodes**
3. Search for **Wan Vace Auto Joiner**
4. Install and restart ComfyUI

> If the node is not listed yet, use manual install while the ComfyUI-Manager registry PR is pending.

---

### Option 2 — Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Rhovanx/wan_vace_auto_joiner.git
```

Restart ComfyUI.

---

## The Three Nodes

| Node | Display Name | Runs | Purpose |
|-----|-------------|------|--------|
| `WanVaceAutoJoiner` | WAN VACE Auto Joiner | Inside loop | INIT (first iteration) / PROCESS (subsequent) |
| `WanVaceAutoJoinerSave` | WAN VACE Auto Joiner – Save | Inside loop | Saves VACE output and acts as loop barrier |
| `WanVaceAutoJoinerFinalize` | WAN VACE Auto Joiner – Finalize | After loop | Outputs all frames (NO VACE) |

---

## Critical Concept: the `value1` Barrier

ComfyUI Easy-Use For Loop must **wait for VACE to fully finish** before advancing to the next iteration.

This node set enforces that by:
- Using `value1` (wildcard `*`) as the loop dependency
- Making the **Save** node depend on **VAE Decode output**
- Feeding `value1` back into `initial_value1`

This guarantees:
- No early loop exit
- No race conditions
- No skipped clips

---

## Required Connection Pattern (IMPORTANT)

```
For Loop Start ─────────────► For Loop End (flow)
        │
        ├── value1 ─► Save ─► initial_value1
        │
        └── index ─► Auto Joiner
                           │
                           ▼
                      WAN VACE
                           │
                       VAE Decode
                           │
                           └──► Save
After loop:
For Loop End (value1) ─────► Finalize
```

---

## Workflow Setup

### 1️⃣ Prepare your input directory

Clips must follow this naming format:

```
clip_00001.mp4
clip_00002.mp4
clip_00003.mp4
```

(The prefix is configurable in the Auto Joiner node.)

---

### 2️⃣ Set loop count correctly

If you have **N videos**:

```
For Loop Start → total = N - 1
```

Example:
- 4 videos → total = 3

---

### 3️⃣ Run once

Queue the workflow **one time** — the loop handles everything automatically.

---

## Troubleshooting

### Loop exits early
- Ensure `value1` passes **through the Save node**
- Do **not** use FLOW_CONTROL for the barrier

### Finalize runs too early
- Confirm Finalize is connected **after** For Loop End
- Do not place Finalize inside the loop

### Missing temp directory
- Auto Joiner must run at least once
- Check directory path and permissions

### Videos not detected
- Ensure filenames match `prefix_00001.mp4`
- Confirm numbering starts at `00001`

---

## FAQ

**Why not FLOW_CONTROL?**  
FLOW_CONTROL does not block asynchronous WAN VACE execution.  
`value1` enforces true dependency completion.

**Why does Finalize not run VACE?**  
VACE is only required for transitions.  
Finalize only assembles already-generated frames.

**Can I use this without Easy-Use?**  
No. The loop system depends on Easy-Use For Loop.

---

## License

MIT
