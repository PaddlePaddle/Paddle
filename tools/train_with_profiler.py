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
Paddle fragmentation profiling with a real training loop.

Runs a small ResNet-like model, collects fragmentation snapshots at key points,
and reports all metrics. Supports both default (non-VMM) and VMM mode.

Usage:
  # Experiment A: default allocator
  python tools/train_with_profiler.py

  # Experiment B: VMM allocator
  FLAGS_use_virtual_memory_auto_growth=true python tools/train_with_profiler.py

  # With options
  python tools/train_with_profiler.py --steps 50 --batch_size 32 --amp
"""

import argparse

import paddle
from paddle import nn
from paddle.device.cuda import gpu_frag_profiler as fp

# ---- Small ResNet-like model (keeps GPU memory usage reasonable) ----


class BasicBlock(nn.Layer):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_ch, out_ch, 3, stride=stride, padding=1, bias_attr=False
        )
        self.bn1 = nn.BatchNorm2D(out_ch)
        self.conv2 = nn.Conv2D(out_ch, out_ch, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_ch, out_ch, 1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_ch),
            )

    def forward(self, x):
        out = paddle.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return paddle.nn.functional.relu(out)


class SmallResNet(nn.Layer):
    """ResNet-18 style, channels halved to keep memory small."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 32, 3, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = paddle.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = paddle.flatten(x, 1)
        return self.fc(x)


def train_loop(args):
    """Run training and collect snapshots."""
    paddle.device.set_device("gpu:0")
    snaps = []

    # Snapshot: before model creation
    snaps.append(fp.snapshot("before_model"))

    model = SmallResNet(num_classes=100)
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=model.parameters()
    )
    loss_fn = nn.CrossEntropyLoss()
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if args.amp else None

    snaps.append(fp.snapshot("after_model"))

    # Synthetic data
    for step in range(1, args.steps + 1):
        x = paddle.randn([args.batch_size, 3, 32, 32])
        y = paddle.randint(0, 100, [args.batch_size])

        if args.amp:
            with paddle.amp.auto_cast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        optimizer.clear_grad()

        # Collect snapshots at key points
        if step == 1:
            snaps.append(fp.snapshot("step_1"))
        elif step == args.steps // 2:
            snaps.append(fp.snapshot(f"step_{step}"))
        elif step == args.steps:
            snaps.append(fp.snapshot(f"step_{step}"))

    # Probe max batch
    def run_one_batch(batch_size=1):
        x = paddle.randn([batch_size, 3, 32, 32])
        with paddle.no_grad():
            model(x)
        paddle.device.synchronize()

    paddle.device.cuda.empty_cache()
    max_bs = fp.probe_max_batch(
        run_one_batch, start=args.batch_size, max_try=2048
    )
    snaps.append(fp.snapshot(f"max_bs={max_bs}"))

    return snaps, max_bs


def main():
    parser = argparse.ArgumentParser(
        description="Paddle frag profiler experiment"
    )
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    args = parser.parse_args()

    vmm = paddle.get_flags("FLAGS_use_virtual_memory_auto_growth").get(
        "FLAGS_use_virtual_memory_auto_growth", False
    )
    print("=== Paddle Frag Profiler ===")
    print(f"    VMM mode: {vmm}")
    print(f"    AMP: {args.amp}")
    print(f"    Steps: {args.steps}, Batch: {args.batch_size}")

    snaps, max_bs = train_loop(args)

    fp.report(snaps)
    print(f"\n>>> Max inference batch size: {max_bs}")
    print(f">>> Allocator: {'VMM' if vmm else 'AutoGrowthBestFit'}")


if __name__ == "__main__":
    main()
