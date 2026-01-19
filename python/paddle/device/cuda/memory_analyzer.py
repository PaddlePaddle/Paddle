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

import numpy as np

import paddle
from paddle.base import core

# --- Constants ---
KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024


# --- Formatting Helpers ---
def format_size(size_bytes):
    if size_bytes == 0:
        return "0 B"
    if size_bytes < MB:
        return f"{size_bytes / KB:.2f} KB"
    if size_bytes < GB:
        return f"{size_bytes / MB:.2f} MB"
    return f"{size_bytes / GB:.2f} GB"


def print_table(title, headers, rows):
    if not rows:
        return
    # Calculate widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    col_widths = [w + 2 for w in col_widths]

    # Build lines
    row_fmt = "|" + "|".join([f"{{:^{w}}}" for w in col_widths]) + "|"
    header_sep = "+" + "+".join(["=" * w for w in col_widths]) + "+"
    inner_sep = "+" + "+".join(["-" * w for w in col_widths]) + "+"

    print(f"\n### {title}")
    print(header_sep)
    print(
        "|" + "|".join([f"{h:^{w}}" for h, w in zip(headers, col_widths)]) + "|"
    )
    print(header_sep)

    for i, row in enumerate(rows):
        print(row_fmt.format(*[str(c) for c in row]))
        if (
            title == "Block Size Distribution"
            and (i + 1) % 2 == 0
            and i != len(rows) - 1
        ):
            print(inner_sep)
        elif title != "Block Size Distribution":
            print(inner_sep)
    if title == "Block Size Distribution":
        print(header_sep)


class MemoryAnalysisTool:
    def __init__(self):
        raise TypeError("Utility class should not be instantiated.")

    @classmethod
    def vmm_max_free_size(
        self, device_id: int | None = None
    ) -> tuple[int, int]:
        name = 'paddle.device.cuda.vmm_max_free_size'
        if not (core.is_compiled_with_cuda()):
            raise ValueError(
                f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
            )
        device_id = (
            device_id
            if device_id is not None
            else core.get_cuda_current_device_id()
        )
        return core.vmm_max_free_size(device_id)

    @classmethod
    def vmm_free_block_info(
        self,
        device_id: int | None = None,
    ) -> list[list[tuple[int, int]]]:
        name = 'paddle.device.cuda.vmm_free_block_info'
        if not (core.is_compiled_with_cuda()):
            raise ValueError(
                f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
            )
        device_id = (
            device_id
            if device_id is not None
            else core.get_cuda_current_device_id()
        )
        return core.vmm_free_block_info(device_id)

    @classmethod
    def vmm_all_block_info(
        self,
        device_id: int | None = None,
    ) -> list[list[tuple[int, int, bool]]]:
        name = 'paddle.device.cuda.vmm_all_block_info'
        if not (core.is_compiled_with_cuda()):
            raise ValueError(
                f"The API {name} is not supported in CPU-only PaddlePaddle. Please reinstall PaddlePaddle with GPU support to call this API."
            )
        device_id = (
            device_id
            if device_id is not None
            else core.get_cuda_current_device_id()
        )
        return core.vmm_all_block_info(device_id)

    @classmethod
    def memory_summary(self, device_id: int | None = None) -> None:
        device_id = (
            device_id
            if device_id is not None
            else core.get_cuda_current_device_id()
        )
        nvidia_smi_AVAILABLE = False
        try:
            # import nvidia_smi, pip install nvidia-ml-py3
            import nvidia_smi

            nvidia_smi_AVAILABLE = True
        except ImportError:
            nvidia_smi_AVAILABLE = False

        THRESHOLDS = [
            1 * MB,
            10 * MB,
            50 * MB,
            100 * MB,
            200 * MB,
            400 * MB,
            600 * MB,
            800 * MB,
            1 * GB,
            2 * GB,
            3 * GB,
        ]
        RANGE_HEADERS = [
            "[0B,1M)",
            "[1M,10M)",
            "[10M,50M)",
            "[50M,100M)",
            "[100M,200M)",
            "[200M,400M)",
            "[400M,600M)",
            "[600M,800M)",
            "[800M,1G)",
            "[1G,2G)",
            "[2G,3G)",
            "[3G,+INF)",
        ]

        allocator_lists = self.vmm_all_block_info(device_id=device_id)
        # --- Feature 1: Global Summary with NVML & Rates ---

        # 1.1 Get Paddle Stats
        mem_allocated = paddle.device.cuda.memory_allocated()
        max_mem_allocated = paddle.device.cuda.max_memory_allocated()
        mem_reserved = paddle.device.cuda.memory_reserved()
        max_mem_reserved = paddle.device.cuda.max_memory_reserved()

        # 1.2 Calculate Rates (Utilization of the Reserved Pool)
        # Rate = How much of the reserved pool is actually holding tensor data?
        max_alloc_rate = (
            ((mem_reserved - max_mem_allocated) / mem_reserved)
            if mem_reserved > 0
            else 0.0
        )

        # 1.3 Get Physical Usage via nvidia_smi
        phy_used_str = "N/A"
        if nvidia_smi_AVAILABLE:
            try:
                nvidia_smi.nvmlInit()

                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                phy_used_str = format_size(info.used)
                phy_total_str = format_size(info.total)
                # nvidia_smi.nvmlShutdown() # Optional, depends on lifecycle
            except Exception as e:
                phy_used_str = "Err"
                phy_total_str = "Err"
        else:
            print(
                "Place install nvidia-smi to check real memory usage, pip install command: `pip install nvidia-ml-py3`"
            )
            phy_used_str = "No nvidia_smi"
            phy_total_str = "No nvidia_smi"

        global_headers = [
            "Allocators",
            "Allocated",
            "Max Alloc",
            "Reserved",
            "Max Reserved",
            "Max Frag Rate",
            "Phy GPU Used / Total",
        ]

        global_rows = [
            [
                len(allocator_lists),
                format_size(mem_allocated),
                format_size(max_mem_allocated),
                format_size(mem_reserved),
                format_size(max_mem_reserved),
                f"{max_alloc_rate:.2%}",
                phy_used_str + ' / ' + phy_total_str,
            ]
        ]

        print_table("Global Memory Snapshot", global_headers, global_rows)

        # --- 2. Allocator Analysis ---
        summary_rows = []
        dist_rows = []

        for idx, blocks in enumerate(allocator_lists):
            allocator_name = f"Allocator_{idx}"

            # A. Basic Counting
            total_blocks = len(blocks)
            free_blocks = 0
            total_size = 0
            free_size = 0
            max_free_size = 0
            max_used_size = 0
            buckets = [[0, 0] for _ in range(len(RANGE_HEADERS))]

            for size, addr, is_free in blocks:
                total_size += size
                if is_free:
                    free_blocks += 1
                    free_size += size
                    max_free_size = max(max_free_size, size)
                else:
                    max_used_size = max(max_used_size, size)

                # Bucket Mapping
                b_idx = len(THRESHOLDS)
                for i, t in enumerate(THRESHOLDS):
                    if size < t:
                        b_idx = i
                        break
                buckets[b_idx][0 if is_free else 1] += 1

            used_blocks = total_blocks - free_blocks
            used_size = total_size - free_size

            # B. Summary Row (Total -> Used -> Free)
            summary_rows.append(
                [
                    allocator_name,
                    total_blocks,
                    used_blocks,
                    free_blocks,
                    format_size(total_size),
                    format_size(used_size),
                    format_size(free_size),
                    format_size(max_used_size),
                    format_size(max_free_size),
                ]
            )

            # D. Distribution Rows
            dist_rows.append(
                [allocator_name, "Free Blocks"] + [b[0] for b in buckets]
            )
            dist_rows.append(
                [allocator_name, "Used Blocks"] + [b[1] for b in buckets]
            )

        # --- 3. Render Outputs ---
        sum_headers = [
            "ID",
            "Tot Blks",
            "Used Blks",
            "Free Blks",
            "Tot Size",
            "Used Size",
            "Free Size",
            "Max Used",
            "Max Free",
        ]
        print_table("Allocator Summary Statistics", sum_headers, summary_rows)

        dist_headers = ["Allocator ID", "Block Type", *RANGE_HEADERS]
        print_table("Block Size Distribution", dist_headers, dist_rows)

    @classmethod
    def allocate_record_table(self, data, output_filepath: str = ""):
        if not data:
            print("No data to display.")
            return

        print(f"Record data size: {len(data)}, start printing...")
        headers = [
            'Allocator_Instance',
            'Is_Allocate',
            'Seq_ID',
            'Req_Size',
            'Cur_Alloc',
            'Cur_Rsrv',
        ]
        formatted_row = []
        all_lines = []
        all_lines.append("\t".join(headers))
        for row in data:
            formatted_row = [
                str(row[0]),
                "Allocate" if row[1] else "Free",
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
            ]
            line = "\t".join(formatted_row)
            all_lines.append(line)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_lines))
            print(f"Data successfully written to: {output_filepath}")
        except OSError as e:
            print(f"Error writing to file {output_filepath}: {e}")

    @classmethod
    def allocate_record_plot(self, data, save_path: str = ""):
        try:
            import matplotlib.pyplot as plt
            from matplotlib import ticker
        except ImportError:
            raise ImportError(
                "matplotlib is required but not installed. Please install it using: pip install matplotlib"
            )

        if not data:
            print("No data to plot.")
            return
        print(f"Record data size: {len(data)}, start plotting...")
        data_np = np.array(data)
        is_allocate = data_np[:, 1]
        filter_mask = is_allocate == 1
        data_np = data_np[filter_mask]
        allocator_instance = data_np[:, 0]  # allocator_instance not used
        ids = data_np[:, 2]
        sizes = data_np[:, 3]
        allocated = data_np[:, 4]
        reserved = data_np[:, 5]

        LOG_START_VALUE = 1
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(16, 10),
            dpi=120,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0},
        )

        # allocated event plot
        ax1.plot(
            ids, sizes, color='#2ca02c', linestyle='-', linewidth=1, alpha=0.3
        )
        ax1.scatter(
            ids,
            sizes,
            color='#2ca02c',
            s=60,
            alpha=1.0,
            edgecolors='white',
            linewidth=0.5,
            label='Request Size',
            zorder=5,
        )

        ax1.set_ylabel(
            'Request Size (Linear Scale)',
            fontsize=12,
            fontweight='bold',
            labelpad=10,
        )
        ax1.set_title(
            'Paddle GPU Memory Allocation Analysis',
            fontsize=16,
            fontweight='bold',
            pad=20,
        )

        ax1.set_ylim(bottom=LOG_START_VALUE)
        ax1.tick_params(axis='x', length=0)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # memory allocated, reserved plot
        ax2.plot(
            ids,
            reserved,
            color='#d62728',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
            label='Reserved (Pool)',
        )
        ax2.fill_between(ids, 0, reserved, color='#d62728', alpha=0.1)
        ax2.plot(
            ids,
            allocated,
            color='#1f77b4',
            linestyle='-',
            linewidth=2,
            alpha=0.9,
            label='Allocated (Used)',
        )
        ax2.fill_between(ids, 0, allocated, color='#1f77b4', alpha=0.15)

        ax2.invert_yaxis()
        ax2.set_ylim(reserved.max() * 3.0, LOG_START_VALUE)
        # ax2.set_yscale('symlog', linthresh=1024 * 1024)
        ax2.set_ylabel(
            'Pool Status (Inverted)',
            fontsize=11,
            fontweight='bold',
            labelpad=10,
        )

        ax2.set_xlabel('')
        ax2.tick_params(axis='x', which='both', length=0)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # y axis setting 0
        def y_axis_formatter(x, pos):
            val = abs(x)
            if val <= LOG_START_VALUE * 1.5:
                return '0'
            return format_size(val).replace(" ", "")

        formatter = ticker.FuncFormatter(y_axis_formatter)
        ax1.yaxis.set_major_formatter(formatter)
        ax2.yaxis.set_major_formatter(formatter)

        for ax in [ax1, ax2]:
            current_ticks = ax.get_yticks().tolist()
            if LOG_START_VALUE not in current_ticks:
                current_ticks.append(LOG_START_VALUE)
            ax.set_yticks(sorted(current_ticks))

        # axis setting
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
            ax.tick_params(
                axis='both', which='major', colors='black', width=1.0, length=5
            )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc='upper right',
            fontsize=10,
            frameon=True,
            facecolor='white',
            framealpha=0.9,
            edgecolor='black',
            shadow=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)

        plt.savefig(save_path)
        plt.close()
        print(f"Analysis plot saved to: {save_path}")
