#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch counterpart of train_with_profiler.py for A/B comparison.

Same model architecture, same data shape, same training loop.
Collects PyTorch native memory stats for comparison with Paddle.

Usage:
  python tools/train_with_profiler_torch.py
  python tools/train_with_profiler_torch.py --steps 50 --batch_size 32 --amp
"""

import argparse

import torch
from torch import nn

MB = 1024 * 1024


# ---- Same ResNet architecture as Paddle version ----


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class SmallResNet(nn.Module):
    """Same architecture as the Paddle version."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def snapshot(tag):
    """Collect PyTorch memory stats in a format comparable to Paddle profiler."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    driver_used = total_gpu - free_gpu

    stats = torch.cuda.memory_stats()

    # Internal frag: requested vs actually allocated
    req_bytes = stats.get("requested_bytes.all.allocated", 0)
    alloc_bytes = stats.get("allocated_bytes.all.allocated", 0)
    internal_frag = 1 - req_bytes / max(alloc_bytes, 1)

    # External frag: inactive_split bytes are fragmented free holes
    inactive_bytes = stats.get("inactive_split_bytes.all.current", 0)
    total_free_in_pool = reserved - allocated
    external_frag = (
        inactive_bytes / max(total_free_in_pool, 1)
        if total_free_in_pool > 0
        else 0.0
    )

    # Segment count (like chunk_count in Paddle)
    segment_count = stats.get("segment.all.current", 0)

    return {
        "tag": tag,
        "allocated_mb": allocated / MB,
        "reserved_mb": reserved / MB,
        "pool_util": allocated / reserved if reserved > 0 else 1.0,
        "driver_used_mb": driver_used / MB,
        "hidden_memory_mb": (driver_used - reserved) / MB,
        "hidden_ratio": (driver_used - reserved) / driver_used
        if driver_used > 0
        else 0,
        "external_frag": external_frag,
        "internal_frag": internal_frag,
        "segment_count": segment_count,
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
    }


def _fmt(val, fmt=".1%"):
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def report(snapshots):
    """Print PyTorch memory stats table (aligned with Paddle profiler output)."""
    hdr = (
        f"{'Tag':>16} {'Alloc':>8} {'Rsrvd':>8} {'Driver':>8} {'Hidden':>8} "
        f"{'Pool%':>6} {'HidRt':>6} "
        f"{'ExtFrag':>7} {'IntFrag':>7} {'Segments':>8} {'Retries':>7}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{'-' * len(hdr)}")
    for s in snapshots:
        print(
            f"{s['tag']:>16} "
            f"{s['allocated_mb']:>7.0f}M {s['reserved_mb']:>7.0f}M "
            f"{s['driver_used_mb']:>7.0f}M "
            f"{s['hidden_memory_mb']:>7.0f}M "
            f"{_fmt(s.get('pool_util')):>6} "
            f"{_fmt(s.get('hidden_ratio')):>6} "
            f"{_fmt(s.get('external_frag')):>7} "
            f"{_fmt(s.get('internal_frag')):>7} "
            f"{s['segment_count']:>8} "
            f"{s['num_alloc_retries']:>7}"
        )
    print(f"{sep}\n")


def probe_max_batch(model, device, start=1, max_try=2048):
    """Binary search for max inference batch size before OOM."""
    lo, hi, best = start, max_try, start
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            x = torch.randn(mid, 3, 32, 32, device=device)
            with torch.no_grad():
                model(x)
            torch.cuda.synchronize()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            hi = mid - 1
        finally:
            torch.cuda.empty_cache()
    return best


def train_loop(args):
    """Run training and collect snapshots."""
    device = torch.device("cuda:0")
    snaps = []

    snaps.append(snapshot("before_model"))

    model = SmallResNet(num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() if args.amp else None

    snaps.append(snapshot("after_model"))

    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 100, (args.batch_size,), device=device)

        if args.amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        if step == 1:
            snaps.append(snapshot("step_1"))
        elif step == args.steps // 2:
            snaps.append(snapshot(f"step_{step}"))
        elif step == args.steps:
            snaps.append(snapshot(f"step_{step}"))

    torch.cuda.empty_cache()
    max_bs = probe_max_batch(model, device, start=args.batch_size, max_try=2048)
    snaps.append(snapshot(f"max_bs={max_bs}"))

    return snaps, max_bs


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch frag profiler experiment"
    )
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    args = parser.parse_args()

    print("=== PyTorch Memory Profiler ===")
    print(f"    AMP: {args.amp}")
    print(f"    Steps: {args.steps}, Batch: {args.batch_size}")

    snaps, max_bs = train_loop(args)

    report(snaps)
    print(f"\n>>> Max inference batch size: {max_bs}")
    print(">>> Allocator: CUDACachingAllocator")


if __name__ == "__main__":
    main()
