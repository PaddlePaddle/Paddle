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

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import TypedDict

# --- Type Aliases ---
# "{data_type[::phi::dtype::bfloat16]; data_layout[STRIDED]; place[Place(gpu:0)]; library_type[PLAIN]}"
KernelConfig = dict[str, str]
# "fused_transpose": [
#     "{data_type[uint8_t]; data_layout[ONEDNN]; place[Place(cpu)]; library_type[MKLDNN]}",
#     "{data_type[int8_t]; data_layout[ONEDNN]; place[Place(cpu)]; library_type[MKLDNN]}",
#     "{data_type[::phi::dtype::bfloat16]; data_layout[ONEDNN]; place[Place(cpu)]; library_type[MKLDNN]}",
#     "{data_type[float]; data_layout[ONEDNN]; place[Place(cpu)]; library_type[MKLDNN]}"
# ]
KernelManifest = dict[str, list[KernelConfig]]


class GpuKernelChangeSummary(TypedDict):
    # New kernel names that contain "gpu".
    new_kernels_with_gpu: list[str]
    # Existing kernels that have new GPU configurations added.
    kernels_with_new_gpu_support: list[str]
    # Existing GPU kernels that ignore data_type changes.
    gpu_kernels_with_new_data_types: dict[str, list[str]]


class KernelManifestComparer:
    def __init__(self, baseline_manifest_data: dict[str, list[str]]) -> None:
        self.baseline_manifest: KernelManifest = self._process_raw_manifest(
            baseline_manifest_data
        )
        self._preprocessed_baseline = {
            kernel_name: {
                "configs_set": {tuple(sorted(cfg.items())) for cfg in configs},
                "gpu_signatures": {
                    self._create_config_signature_without_dtype(cfg)
                    for cfg in configs
                    if "gpu" in cfg.get("place", "")
                },
            }
            for kernel_name, configs in self.baseline_manifest.items()
        }

    def _parse_config_string(self, config_str: str) -> KernelConfig:
        pattern = r"([\w:]+)\[([^\]]+)\]"
        matches = re.findall(pattern, config_str)
        return dict(matches)

    def _create_config_signature_without_dtype(
        self, config: KernelConfig
    ) -> tuple[tuple[str, str], ...]:
        signature_items = {
            key: value for key, value in config.items() if key != "data_type"
        }
        return tuple(sorted(signature_items.items()))

    def _process_raw_manifest(
        self, raw_manifest: dict[str, list[str]]
    ) -> KernelManifest:
        processed_manifest: KernelManifest = {}
        for kernel_name, config_strings in raw_manifest.items():
            processed_manifest[kernel_name] = [
                self._parse_config_string(cs) for cs in config_strings
            ]
        return processed_manifest

    def compare(
        self, target_manifest_data: dict[str, list[str]]
    ) -> GpuKernelChangeSummary:
        target_manifest = self._process_raw_manifest(target_manifest_data)

        new_kernels_with_gpu: set[str] = set()
        kernels_with_new_gpu_support: set[str] = set()
        gpu_kernels_with_new_data_types: dict[str, set[str]] = {}

        baseline_kernel_names = set(self._preprocessed_baseline.keys())
        target_kernel_names = set(target_manifest.keys())

        # 1. Find newly added kernels that are GPU-related.
        added_kernel_names = target_kernel_names - baseline_kernel_names
        for kernel_name in added_kernel_names:
            if any(
                "gpu" in cfg.get("place", "")
                for cfg in target_manifest[kernel_name]
            ):
                new_kernels_with_gpu.add(kernel_name)

        # 2. Find changes within existing kernels.
        common_kernel_names = baseline_kernel_names.intersection(
            target_kernel_names
        )
        for kernel_name in common_kernel_names:
            baseline_data = self._preprocessed_baseline[kernel_name]
            target_configs = target_manifest[kernel_name]

            target_configs_set = {
                tuple(sorted(cfg.items())) for cfg in target_configs
            }

            if baseline_data["configs_set"] == target_configs_set:
                continue

            added_config_tuples = (
                target_configs_set - baseline_data["configs_set"]
            )
            added_configs = [dict(t) for t in added_config_tuples]

            baseline_gpu_signatures = baseline_data["gpu_signatures"]
            has_baseline_gpu_support = bool(baseline_gpu_signatures)

            for added_cfg in added_configs:
                if "gpu" in added_cfg.get("place", ""):
                    kernels_with_new_gpu_support.add(kernel_name)
                    added_signature = (
                        self._create_config_signature_without_dtype(added_cfg)
                    )
                    if (
                        has_baseline_gpu_support
                        and added_signature in baseline_gpu_signatures
                    ):
                        new_data_type = added_cfg.get("data_type", "N/A")
                        gpu_kernels_with_new_data_types.setdefault(
                            kernel_name, set()
                        ).add(new_data_type)

        return {
            "new_kernels_with_gpu": sorted(new_kernels_with_gpu),
            "kernels_with_new_gpu_support": sorted(
                kernels_with_new_gpu_support
            ),
            "gpu_kernels_with_new_data_types": {
                k: sorted(v) for k, v in gpu_kernels_with_new_data_types.items()
            },
        }


def cli():
    parser = argparse.ArgumentParser(
        description="Compare GPU kernel configurations between two manifests."
    )
    parser.add_argument(
        "baseline_manifest",
        type=Path,
        help="Path to the baseline manifest JSON file.",
    )
    parser.add_argument(
        "target_manifest",
        type=Path,
        help="Path to the target manifest JSON file.",
    )
    parser.add_argument(
        "--ignore-data-type-changes",
        action="store_true",
        help="If set, ignore data_type changes in GPU kernel comparisons.",
    )
    args = parser.parse_args()
    return args


def main():
    args = cli()
    if not args.baseline_manifest.exists():
        raise ValueError(
            f"Baseline manifest file not found: {args.baseline_manifest.resolve()}"
        )
    if not args.target_manifest.exists():
        raise ValueError(
            f"Target manifest file not found: {args.target_manifest.resolve()}"
        )

    with args.baseline_manifest.open("r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with args.target_manifest.open("r", encoding="utf-8") as f:
        target_data = json.load(f)

    comparer = KernelManifestComparer(baseline_data)
    summary = comparer.compare(target_data)

    has_reportable_changes = False
    if args.ignore_data_type_changes:
        all_gpu_related_changes = set(summary["new_kernels_with_gpu"]) | set(
            summary["kernels_with_new_gpu_support"]
        )
        kernels_with_only_dtype_changes = set(
            summary["gpu_kernels_with_new_data_types"].keys()
        )
        unignored_changes = (
            all_gpu_related_changes - kernels_with_only_dtype_changes
        )
        if unignored_changes:
            has_reportable_changes = True
    else:
        if (
            summary["new_kernels_with_gpu"]
            or summary["kernels_with_new_gpu_support"]
        ):
            has_reportable_changes = True

    if summary["new_kernels_with_gpu"]:
        print("New GPU Kernels Added:")
        for kernel in summary["new_kernels_with_gpu"]:
            print(f"  - {kernel}")

    if summary["kernels_with_new_gpu_support"]:
        print("\nKernels with New GPU Support:")
        for kernel in summary["kernels_with_new_gpu_support"]:
            print(f"  - {kernel}")

    if summary["gpu_kernels_with_new_data_types"]:
        print("\nGPU Kernels with new Data Type:")
        for kernel, data_types in summary[
            "gpu_kernels_with_new_data_types"
        ].items():
            data_types_str = ", ".join(data_types)
            print(f"  - {kernel}: New data types - {data_types_str}")

    if has_reportable_changes:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
