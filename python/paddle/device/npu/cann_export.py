# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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


import json
import os
import subprocess

import paddle


def cann_parse_enabled():
    """
    Automatically parse profiling data for NPU devices using CANN tools.
    """
    prof_root_dir = os.getenv(
        'PROFILER_OUTPUT_DIR', os.path.join(os.getcwd(), 'ascend_profiling')
    )

    if not is_npu_device():
        return

    latest_prof_dir = find_latest_prof_directory(prof_root_dir)
    if not latest_prof_dir:
        print(f"No PROF directories found in {prof_root_dir}.")
        return

    latest_prof_path = os.path.join(prof_root_dir, latest_prof_dir)
    run_msprof_command(latest_prof_path)
    merge_json_files(latest_prof_path)


def is_npu_device():
    """
    Check if the current device is NPU.
    """
    return "npu" in paddle.device.get_device()


def find_latest_prof_directory(prof_root_dir):
    """
    Find the latest PROF_* directory in the profiling root directory.
    """
    try:
        prof_dirs = [
            d for d in os.listdir(prof_root_dir) if d.startswith('PROF_')
        ]
        if not prof_dirs:
            return None
        return max(
            prof_dirs,
            key=lambda d: os.path.getctime(os.path.join(prof_root_dir, d)),
        )
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing {prof_root_dir}: {e!s}")
        return None


def run_msprof_command(prof_dir):
    """
    Run the msprof command to export profiling data.
    """
    msprof_command = f"msprof --export=on --output={prof_dir}"
    print(f"Running msprof command: {msprof_command}")
    try:
        subprocess.run(msprof_command, shell=True, check=True)
        print("Profiling data parsed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running msprof command: {e!s}")


def merge_json_files(prof_dir):
    """
    Merge the JSON files from msprof and paddle, adjusting sort_index to ensure correct event order.
    """
    try:
        msprof_json_path = find_latest_msprof_json(prof_dir)
        if not msprof_json_path:
            print(
                f"No msprof JSON files found in {os.path.join(prof_dir, 'mindstudio_profiler_output')}."
            )
            return

        paddle_json_path = find_latest_paddle_json()
        if not paddle_json_path:
            print("No Paddle JSON files found.")
            return

        msprof_data = load_json(msprof_json_path)
        if not msprof_data:
            print(f"msprof JSON file {msprof_json_path} is empty or invalid.")
            return

        paddle_data = load_json(paddle_json_path)
        if not paddle_data:
            print(f"Paddle JSON file {paddle_json_path} is empty or invalid.")
            return

        paddle_events = paddle_data.get('traceEvents', [])
        msprof_events = msprof_data if isinstance(msprof_data, list) else []

        adjust_paddle_sort_index(paddle_events, msprof_events)

        merged_data = {'traceEvents': paddle_events + msprof_events}
        output_json_path = os.path.join(prof_dir, 'merged_trace.json')
        save_json(merged_data, output_json_path)
        print(f"Merged JSON file saved to {output_json_path}")

    except Exception as e:
        print(f"Error during JSON merge: {e!s}")


def find_latest_msprof_json(prof_dir):
    """
    Find the latest msprof JSON file in the specified directory.
    """
    try:
        msprof_output_dir = os.path.join(prof_dir, 'mindstudio_profiler_output')
        msprof_json_files = [
            f
            for f in os.listdir(msprof_output_dir)
            if f.startswith('msprof_') and f.endswith('.json')
        ]
        if not msprof_json_files:
            return None
        return os.path.join(
            msprof_output_dir,
            max(
                msprof_json_files,
                key=lambda f: os.path.getmtime(
                    os.path.join(msprof_output_dir, f)
                ),
            ),
        )
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error finding msprof JSON files: {e!s}")
        return None


def find_latest_paddle_json():
    """
    Find the latest Paddle JSON file in the current working directory.
    """
    try:
        paddle_json_dir = os.path.join(os.getcwd(), 'profiler_demo')
        paddle_json_files = [
            f
            for f in os.listdir(paddle_json_dir)
            if f.endswith('.paddle_trace.json')
        ]
        if not paddle_json_files:
            return None
        return os.path.join(
            paddle_json_dir,
            max(
                paddle_json_files,
                key=lambda f: os.path.getmtime(
                    os.path.join(paddle_json_dir, f)
                ),
            ),
        )
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error finding Paddle JSON files: {e!s}")
        return None


def load_json(file_path):
    """
    Load a JSON file from the specified path.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file {file_path}: {e!s}")
        return None
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error opening JSON file {file_path}: {e!s}")
        return None


def save_json(data, file_path):
    """
    Save the given data to a JSON file at the specified path.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except (OSError, PermissionError) as e:
        print(f"Error saving JSON file {file_path}: {e!s}")


def adjust_paddle_sort_index(paddle_events, msprof_events):
    """
    Adjust the sort_index of Paddle events to ensure they appear before MSProf events.
    """
    min_sort_index_msprof = min(
        (
            event.get('args', {}).get('sort_index', float('inf'))
            for event in msprof_events
            if 'args' in event
        ),
        default=0,
    )

    min_sort_index_paddle = min(
        (
            event.get('args', {}).get('sort_index', float('inf'))
            for event in paddle_events
            if 'args' in event
        ),
        default=0,
    )

    adjustment_value = min_sort_index_msprof - min_sort_index_paddle - 1

    for event in paddle_events:
        if 'args' in event and 'sort_index' in event['args']:
            event['args']['sort_index'] += adjustment_value
