"""Train the small openWakeWord classifier head on the features from features.py and export it as ONNX.

Runs in the separate torch venv (data/wakeword/venv), so the app's own environment never needs torch:

    data/wakeword/venv/Scripts/python scripts/wakeword/train.py [--dim 64] [--epochs 60]

The network is openWakeWord's own "dnn" head (Flatten -> Linear -> LN -> ReLU -> [Linear -> LN -> ReLU] -> Linear ->
Sigmoid) on a 16x96 embedding window; export gives a 1x16x96 -> 1x1 ONNX that EV's WakeWord class loads unchanged.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
FEATS = ROOT / "data" / "wakeword" / "features"
OUT = ROOT / "models" / "openwakeword"


class Head(nn.Module):
    def __init__(self, dim: int, shape: tuple[int, int] = (16, 96)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(shape[0] * shape[1], dim), nn.ReLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(),
            nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def metrics(m: Head, X: torch.Tensor, y: np.ndarray, kinds: np.ndarray, thr: float = 0.5) -> dict:
    m.eval()
    with torch.no_grad():
        p = m(X).squeeze(1).numpy()
    pos = y == 1
    out = {"recall": float((p[pos] >= thr).mean()), "fp": float((p[~pos] >= thr).mean())}
    for k in sorted(set(kinds)):
        sel = kinds == k
        out[k] = float((p[sel] >= thr).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64, help="hidden width; 64 is ~110k params (~450 KB)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--neg-weight", type=float, default=4.0, help="final weight on negatives (ramped up from 1)")
    ap.add_argument("--max-fp", type=float, default=0.002, help="val false-positive rate allowed when picking a checkpoint")
    ap.add_argument("--name", default="hey_evie")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr, va = np.load(FEATS / "train.npz"), np.load(FEATS / "val.npz")
    Xtr, ytr = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])
    Xva, yva, kva = torch.from_numpy(va["X"]), va["y"], va["kinds"]
    print(f"train {tuple(Xtr.shape)} ({int(ytr.sum())} pos)   val {tuple(Xva.shape)} ({int(yva.sum())} pos)")

    m = Head(a.dim)
    print(f"{sum(p.numel() for p in m.parameters()):,} parameters")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * (len(Xtr) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
    best, best_state, snaps = None, None, []

    for ep in range(a.epochs):
        m.train()
        negw = 1.0 + (a.neg_weight - 1.0) * ep / max(1, a.epochs - 1)         # push false triggers down over time
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            x, y = Xtr[idx], ytr[idx]
            x = x + 0.02 * torch.randn_like(x) * x.std()                       # feature jitter
            p = m(x).squeeze(1).clamp(1e-6, 1 - 1e-6)
            w = torch.where(y == 1, torch.ones_like(y), torch.full_like(y, negw))
            loss = (nn.functional.binary_cross_entropy(p, y, reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        r = metrics(m, Xva, yva, kva)
        print(f"ep {ep + 1:3d}  loss {loss.item():.4f}  recall {r['recall']:.3f}  fp {r['fp']:.4f}")
        if ep >= a.epochs // 2:
            snaps.append(copy.deepcopy(m.state_dict()))
        if r["fp"] <= a.max_fp and (best is None or r["recall"] > best):
            best, best_state = r["recall"], copy.deepcopy(m.state_dict())

    # average the late checkpoints (what openWakeWord's auto_train does too), fall back to the best single one
    avg = {k: sum(s[k] for s in snaps) / len(snaps) for k in snaps[0]}
    m.load_state_dict(avg)
    ra = metrics(m, Xva, yva, kva)
    if best_state is not None and not (ra["fp"] <= a.max_fp and ra["recall"] >= best - 0.01):
        m.load_state_dict(best_state)
        print(f"using best single checkpoint (recall {best:.3f})")
    else:
        print("using averaged late checkpoints")
    final = metrics(m, Xva, yva, kva)
    print("\nvalidation @0.5 (window level)")
    for k, v in final.items():
        print(f"  {k:10s} {v:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.name}.onnx"
    m.eval()
    torch.onnx.export(m.cpu(), torch.rand(1, 16, 96), str(path), opset_version=13, dynamo=False,
                      input_names=["input"], output_names=[a.name])
    print(f"\nsaved {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
