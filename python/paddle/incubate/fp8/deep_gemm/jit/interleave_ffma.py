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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import argparse
import mmap
import os
import re
import subprocess

from ..utils import get_cuda_home


def run_cuobjdump(file_path):
    CUDA_HOME = get_cuda_home()
    command = [f"{CUDA_HOME}/bin/cuobjdump", "-sass", file_path]
    result = subprocess.run(command, capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"Command failed with return code {result.returncode}: {result.stderr}"
    return result.stdout


def extract_ffma(sass):
    """
    Extract FFMA segments from SASS code.

    Args:
        sass (str): The SASS code as a string.

    Returns:
        list: A list of tuples, each containing the function and architecture name along with the FFMA segments.
    """
    lines = sass.splitlines()
    collected = []
    current = []

    arch_name, func_name = "N/A", "N/A"
    skip_next_line = False

    for line in lines:
        # Extract architecture name
        if "code for" in line:
            arch_name = line.strip().split("code for", 1)[-1].strip()

        # Extract function name using regex
        elif "Function :" in line:
            match = re.search(r"Function : (\S+)", line)
            if match:
                func_name = match.group(1)

        # Collect FFMA segments
        elif "FFMA" in line:
            current.append(line)
            skip_next_line = True
        elif skip_next_line:
            current.append(line)
            skip_next_line = False
        else:
            if len(current) >= 16:
                assert (
                    len(current) % 2 == 0
                ), "FFMA segments count should be even"
                collected.append((f"{arch_name}::{func_name}", current))
            current = []

    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(f"Found {len(collected)} FFMA segments")
    return collected


def extract_hex_from_line(line):
    match = re.search(r"/\*\s*(0x[0-9a-fA-F]+)\s*\*/", line)
    assert match
    return int(match.group(1), 16)


def validate(m, offset, le_bytes, num_lines):
    assert len(le_bytes) == num_lines // 2
    assert m[offset : offset + 16] == le_bytes[0]
    for i in range(1, num_lines // 2):
        if m[offset + i * 16 : offset + i * 16 + 16] != le_bytes[i]:
            return False
    return True


def parse_registers(line):
    line = re.sub(r"/\*.*?\*/", "", line)
    line = line.replace(";", "")
    tokens = line.strip().split(",")
    registers = []
    for token in tokens:
        token = token.strip()
        words = token.split()
        for word in words:
            if word.startswith("R"):
                reg = word.split(".")[0]
                registers.append(reg)
    return registers


def modify_segment(m, name, ffma_lines):
    num_lines = (len(ffma_lines) * 9 // 16) // 2 * 2
    assert num_lines % 2 == 0

    le_bytes, new_le_bytes = [], []
    reused_list = []
    dst_reg_set = set()
    last_reused, last_dst_reg = False, ""
    num_changed = 0
    for i in range(num_lines // 2):
        dst_reg = parse_registers(ffma_lines[i * 2])[-2]
        low_line, high_line = ffma_lines[i * 2], ffma_lines[i * 2 + 1]
        low_hex, high_hex = extract_hex_from_line(
            low_line
        ), extract_hex_from_line(high_line)
        le_bytes.append(
            low_hex.to_bytes(8, "little") + high_hex.to_bytes(8, "little")
        )
        reused = (high_hex & 0x0800000000000000) != 0
        if reused:
            is_first_occurred = dst_reg not in dst_reg_set
            if is_first_occurred or (last_reused and dst_reg == last_dst_reg):
                # Modify the `reuse` and `yield` bits
                assert high_hex & 0x0800200000000000, f"{hex(high_hex)}"
                high_hex ^= 0x0800200000000000
                reused = False
                num_changed += 1
            else:
                reused_list.append(i)
        dst_reg_set.add(dst_reg)
        new_le_bytes.append(
            low_hex.to_bytes(8, "little") + high_hex.to_bytes(8, "little")
        )
        last_reused, last_dst_reg = reused, dst_reg
    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(
            f" > segment `{name}` new reused list ({num_changed} changed): {reused_list}"
        )

    # Find the offset
    offsets = []
    offset = m.find(le_bytes[0])
    while offset != -1:
        offsets.append(offset)
        offset = m.find(le_bytes[0], offset + 1)
    offsets = list(
        filter(lambda x: validate(m, x, le_bytes, num_lines), offsets)
    )

    # Replace with `new_le_bytes`
    for offset in offsets:
        for i in range(num_lines // 2):
            m[offset + i * 16 : offset + i * 16 + 16] = new_le_bytes[i]


def process(path):
    if os.getenv("DG_PRINT_REG_REUSE", None):
        print(f"Processing {path}")
    output = run_cuobjdump(path)
    segments = extract_ffma(output)
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
        for segment in segments:
            modify_segment(mm, *segment)
        mm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interleave FFMA reg reuse")
    parser.add_argument("--so", help="Path to the SO file")
    args = parser.parse_args()

    process(args.so)
