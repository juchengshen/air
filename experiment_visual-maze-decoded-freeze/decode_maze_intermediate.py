#!/usr/bin/env python3
"""
Decode AIR-1net intermediate output on Maze Hard at each recursive step (L/H).
Uses GPU by default. Saves all PNGs to AIR_code/experiment_visual-maze-decoded-freeze/maze_20k_epochs/.

Run from AIR_code repo root:
  python experiment_visual-maze-decoded-freeze/decode_maze_intermediate.py
"""
from __future__ import annotations

import os
import sys
import yaml

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["figure.titleweight"] = "bold"
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AIR_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "maze_20k_epochs")
OUT_DIR = os.environ.get("AIR_MAZE_OUT_DIR", DEFAULT_OUT_DIR)
if not os.path.isabs(OUT_DIR):
    OUT_DIR = os.path.join(AIR_ROOT, OUT_DIR)
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

if AIR_ROOT not in sys.path:
    sys.path.insert(0, AIR_ROOT)
os.chdir(AIR_ROOT)

os.environ["DISABLE_COMPILE"] = "1"

from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.air.air_1net_Lx_H import AIR_Inner, AIRInnerCarry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE = 30

# Token mapping (from build_maze_dataset.py CHARSET "# SGo"):
#   0 = PAD, 1 = # (wall), 2 = space (path), 3 = S (start), 4 = G (goal), 5 = o (solution)
MAZE_CMAP = ListedColormap([
    "#808080",  # 0: PAD   – gray
    "#1a1a1a",  # 1: #     – black (wall)
    "#ffffff",  # 2: space – white (open path)
    "#22cc22",  # 3: S     – green (start)
    "#dd2222",  # 4: G     – red   (goal)
    "#3388ff",  # 5: o     – blue  (solution path)
])


def run_inner_with_tracing(inner: AIR_Inner, carry, batch: dict):
    """Run the inner forward step-by-step, decoding z_H and z_L after each L and H update."""
    seq_info = {"cos_sin": inner.rotary_emb() if hasattr(inner, "rotary_emb") else None}
    input_embeddings = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    plen = inner.puzzle_emb_len
    H_cycles = inner.config.H_cycles
    L_cycles = inner.config.L_cycles

    z_H = carry.z_H.clone()
    z_L = carry.z_L.clone()
    traces = []

    def decode(z):
        logits = inner.lm_head(z)[:, plen:]
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().numpy()

    with torch.no_grad():
        for H_step in range(H_cycles):
            for L_step in range(L_cycles):
                if not ((H_step == H_cycles - 1) and (L_step == L_cycles - 1)):
                    z_L = inner.f_level(z_L, z_H + input_embeddings, **seq_info)
                    traces.append({
                        "stage": "L", "H_step": H_step, "L_step": L_step,
                        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
                        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
                    })
            if not (H_step == H_cycles - 1):
                z_H = inner.f_level(z_H, z_L, **seq_info)
                traces.append({
                    "stage": "H", "H_step": H_step, "L_step": None,
                    "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
                    "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
                })

    z_L = inner.f_level(z_L, z_H + input_embeddings, **seq_info)
    traces.append({
        "stage": "L", "H_step": H_cycles - 1, "L_step": L_cycles - 1,
        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
    })
    z_H = inner.f_level(z_H, z_L, **seq_info)
    traces.append({
        "stage": "H", "H_step": H_cycles - 1, "L_step": None,
        "pred_H": decode(z_H).reshape(-1, GRID_SIZE, GRID_SIZE),
        "pred_L": decode(z_L).reshape(-1, GRID_SIZE, GRID_SIZE),
    })

    new_carry = AIRInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
    logits = inner.lm_head(z_H)[:, plen:]
    q_logits = inner.q_head(z_H[:, 0]).to(torch.float32)
    q_halt, q_continue = q_logits[..., 0], q_logits[..., 1]
    return traces, new_carry, logits, q_halt, q_continue


def plot_maze(ax, grid, title=""):
    """Plot a maze grid using token ids directly (0=PAD .. 5=solution)."""
    grid = np.asarray(grid, dtype=np.int32)
    ax.imshow(grid, cmap=MAZE_CMAP, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def main():
    print("Device:", DEVICE)
    default_ckpt = os.path.join(
        AIR_ROOT,
        "checkpoints",
        "Maze",
        "air_Lx_H_bs768_lr1e4_20k_epochs",
        "step_26040",
    )
    CHECKPOINT_FILE = os.environ.get("AIR_MAZE_CKPT_PATH", default_ckpt)
    if not os.path.isabs(CHECKPOINT_FILE):
        CHECKPOINT_FILE = os.path.join(AIR_ROOT, CHECKPOINT_FILE)
    CHECKPOINT_FILE = os.path.abspath(CHECKPOINT_FILE)
    CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_FILE)
    CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "all_config.yaml")
    batch_index = int(os.environ.get("AIR_MAZE_BATCH_INDEX", "0"))

    if not os.path.isfile(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_FILE}")
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found next to checkpoint: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    print("Config:", config.run_name)

    train_loader, train_metadata = create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", 0, 1, test_set_mode=True, epochs_per_iter=1,
        global_batch_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)
    if DEVICE.type == "cpu":
        train_state.model.to(DEVICE)
    ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    try:
        train_state.model.load_state_dict(ckpt, assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ckpt.items()},
            assign=True
        )
    train_state.model.eval()

    air = train_state.model.model
    inner = air.inner

    eval_iter = iter(eval_loader)
    for _ in range(batch_index + 1):
        _set_name, batch, _batch_size = next(eval_iter)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    batch_size = batch["inputs"].shape[0]

    with torch.device(DEVICE.type):
        carry = air.initial_carry(batch)

    act_step_traces = []
    halt_max_steps = air.config.halt_max_steps

    with torch.inference_mode():
        for act_step in range(halt_max_steps):
            new_inner_carry = air.inner.reset_carry(carry.halted, carry.inner_carry)
            new_steps = torch.where(carry.halted, 0, carry.steps)
            new_current_data = {
                k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                for k, v in carry.current_data.items()
            }
            traces, new_inner_carry, logits, q_halt, q_continue = run_inner_with_tracing(
                inner, new_inner_carry, new_current_data
            )
            act_step_traces.append({"act_step": act_step, "traces": traces, "logits": logits.detach()})
            new_steps = new_steps + 1
            is_last = new_steps >= halt_max_steps
            halted = is_last
            carry = type(carry)(
                inner_carry=new_inner_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data
            )
            if halted.all():
                break

    print(f"Decoded eval batch index {batch_index}.")
    print(f"Ran {len(act_step_traces)} ACT steps. Saving PNGs to {OUT_DIR}")

    inputs_np = batch["inputs"].cpu().numpy().reshape(batch_size, GRID_SIZE, GRID_SIZE)
    labels_np = batch["labels"].cpu().numpy().reshape(batch_size, GRID_SIZE, GRID_SIZE)
    final_pred = torch.argmax(act_step_traces[-1]["logits"], dim=-1).cpu().numpy().reshape(batch_size, GRID_SIZE, GRID_SIZE)

    # 1) Input, solution, final prediction
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_maze(axes[0], inputs_np[0], "Input")
    plot_maze(axes[1], labels_np[0], "Solution")
    plot_maze(axes[2], final_pred[0], "Final pred")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "input_solution_final.png"), dpi=500)
    plt.close(fig)

    # 2) Per-ACT-step z_H and z_L at each recursive (L/H) update
    ncols = 4
    for act_idx, step_data in enumerate(act_step_traces):
        traces = step_data["traces"]
        n = len(traces)
        nrows = (n + ncols - 1) // ncols
        # z_H
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for idx, t in enumerate(traces):
            label = f"{t['stage']} H{t['H_step']}"
            if t["L_step"] is not None:
                label += f" L{t['L_step']}"
            plot_maze(axes[idx], t["pred_H"][0], label)
        for idx in range(len(traces), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f"ACT step {act_idx}: decoded from z_H at each recursive (L/H) update")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"act{act_idx}_zH_per_recursive_step.png"), dpi=500)
        plt.close(fig)
        # z_L
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for idx, t in enumerate(traces):
            label = f"{t['stage']} H{t['H_step']}"
            if t["L_step"] is not None:
                label += f" L{t['L_step']}"
            plot_maze(axes[idx], t["pred_L"][0], label)
        for idx in range(len(traces), len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f"ACT step {act_idx}: decoded from z_L at each recursive (L/H) update")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"act{act_idx}_zL_per_recursive_step.png"), dpi=500)
        plt.close(fig)

    # 3) Last decoded z_H at end of each ACT step (4x4 grid)
    nsteps = len(act_step_traces)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for idx, step_data in enumerate(act_step_traces):
        last_trace = step_data["traces"][-1]
        plot_maze(axes[idx], last_trace["pred_H"][0], f"ACT step {idx}")
    for idx in range(nsteps, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Decoded z_H at end of each ACT step")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "zH_per_act_step.png"), dpi=500)
    plt.close(fig)

    # 4) Last decoded z_L at end of each ACT step (4x4 grid)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for idx, step_data in enumerate(act_step_traces):
        last_trace = step_data["traces"][-1]
        plot_maze(axes[idx], last_trace["pred_L"][0], f"ACT step {idx}")
    for idx in range(nsteps, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Decoded z_L at end of each ACT step")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "zL_per_act_step.png"), dpi=500)
    plt.close(fig)

    print("Done. Outputs:")
    for name in sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".png")):
        print(" ", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
