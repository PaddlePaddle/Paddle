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
test_runner.py

USAGE:
This script is designed to discover and execute pytest test files in parallel across multiple Devices.
It manages test results and handles disabled tests according to configuration files.

FEATURES:
- Discovers Python files starting with 'test_' in the specified path (--path)
- Paddle unit tests are located in the Paddle/test/legacy_test directory, with all test names prefixed by ‘test_’.
- Reads disabled_test.txt from the script directory (supports wildcards and exact names)
- Manages all_tests.txt and tests_result.txt in the script directory
- Executes tests in parallel across available Devices with a timeout of 180s (configurable via --timeout)
- Saves failed or timeout logs to script_directory/failed_logs/<test_file>.log
- Supports skipping float64 tests via --skip-float64 (sets FLAG_SKIP_FLOAT64=1 in subprocess environment)

REQUIREMENTS:
- Python 3.10+
- PaddlePaddle installed
- pytest
- NVIDIA GPU or Custom Device with CUDA
- Optional: Iluvatar GPU with ixsmi tool for GPU detection

USAGE EXAMPLES:
1. Basic usage (run tests in current directory):
   python test_runner.py

2. Run tests in a specific directory:
   python test_runner.py --path /path/to/tests

3. Set custom timeout (default is 180 seconds):
   python test_runner.py --timeout 300

4. Skip float64 tests:
   python test_runner.py --skip-float64

5. Specify GPU devices directly using --devices parameter:
   python test_runner.py --devices 0,1,2,3  # Use GPUs 0, 1, 2, and 3
   python test_runner.py --devices 0,2,4    # Use GPUs 0, 2, and 4
   python test_runner.py --devices 8-15     # Use GPUs 8 through 15 (inclusive)

6. Control which GPUs to use by setting CUDA_VISIBLE_DEVICES:
   export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0 to 3
   python test_runner.py

CONFIGURATION FILES:
- disabled_test.txt: Contains patterns of test files to skip (supports wildcards)
- all_tests.txt: Lists all discovered test files
- tests_result.txt: Tracks test execution status
- failed_logs/: Directory containing logs of failed tests

GPU PARALLEL EXECUTION:
The script automatically detects available GPUs and distributes tests across them.
Each test runs with CUDA_VISIBLE_DEVICES set to a specific GPU index.

GPU Selection Priority:
1. --devices parameter (highest priority)
2. CUDA_VISIBLE_DEVICES environment variable
3. Auto-detection using nvidia-smi or ixsmi
4. Default to single GPU (index 0) if no GPUs detected

If --devices parameter is provided:
- The script will use the specified GPU device IDs directly
- Supports comma-separated lists (e.g., "0,1,2,3") and ranges (e.g., "8-15")
- Tests will be distributed across the specified GPUs

If CUDA_VISIBLE_DEVICES is set (and --devices is not provided):
- The script will only use the GPUs specified in the environment variable
- Tests will be distributed across those specific GPUs
- For example, if CUDA_VISIBLE_DEVICES="0,2,3", tests will run on GPUs 0, 2, and 3

If neither --devices nor CUDA_VISIBLE_DEVICES is set:
- The script will attempt to detect all available NVIDIA GPUs using nvidia-smi
- If nvidia-smi fails, it will try to detect Iluvatar GPUs using ixsmi
- Tests will be distributed across all detected GPUs
- If no GPUs are detected, tests run sequentially on CPU

Note: In subprocesses, the specified GPUs will be remapped to indices 0 to N-1

TEST STATUS:
- [NOT TESTED]: Test discovered but not yet executed
- [PASSED]: Test completed successfully
- [FAILED]: Test failed with errors
- [TIMEOUT]: Test exceeded the time limit

"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import os
import subprocess
import sys

# Status constants
STATUS_NOT_TESTED = "[NOT TESTED]"
STATUS_PASSED = "[PASSED]"
STATUS_FAILED = "[FAILED]"
STATUS_TIMEOUT = "[TIMEOUT]"

# File name constants (located in script directory)
ALL_TESTS_FILE = "all_tests.txt"
TESTS_RESULT_FILE = "tests_result.txt"
DISABLED_FILE = "disabled_test.txt"
FAILED_LOG_DIR = "failed_logs"
TIMEOUT_SECONDS = 180


def script_dir() -> str:
    """Return the absolute path of the directory containing test_runner.py"""
    return os.path.dirname(os.path.abspath(__file__))


def find_test_files(base_path: str) -> list[str]:
    """Discover Python files starting with 'test_', return list of paths relative to base_path"""
    test_files = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.startswith("test_") and f.endswith(".py"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_path)
                test_files.append(rel)
    test_files.sort()
    return test_files


def read_disabled_patterns(conf_dir: str) -> list[str]:
    """Read disabled_test.txt (in script directory)"""
    path = os.path.join(conf_dir, DISABLED_FILE)
    patterns = []
    if not os.path.exists(path):
        return patterns
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def filter_disabled(test_files: list[str], patterns: list[str]) -> list[str]:
    """Filter test files according to disabled_test.txt rules"""
    if not patterns:
        return test_files[:]
    kept = []
    for rel in test_files:
        basename = os.path.basename(rel)
        disabled = False
        for pat in patterns:
            # Support path matching
            if os.sep in pat or "/" in pat:
                normalized_rel = rel.replace(os.sep, "/")
                normalized_pat = pat.replace(os.sep, "/")
                if fnmatch.fnmatch(normalized_rel, normalized_pat):
                    disabled = True
                    break
            else:
                if fnmatch.fnmatch(basename, pat):
                    disabled = True
                    break
        if not disabled:
            kept.append(rel)
    return kept


def load_all_tests_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_all_tests_file(path: str, tests: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{t}\n" for t in tests)


def load_tests_result_file(path: str) -> dict[str, str]:
    """Read tests_result.txt -> {file: status}"""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("["):
                try:
                    end_idx = line.index("]") + 1
                    status = line[:end_idx]
                    rel = line[end_idx:].strip()
                except ValueError:
                    continue
            else:
                status = STATUS_NOT_TESTED
                rel = line.strip()
            if rel:
                results[rel] = status
    return results


def save_tests_result_file(
    path: str, results: dict[str, str], order: list[str]
):
    """Save tests_result.txt"""
    with open(path, "w", encoding="utf-8") as f:
        seen = set()
        for rel in order:
            status = results.get(rel, STATUS_NOT_TESTED)
            f.write(f"{status} {rel}\n")
            seen.add(rel)
        for rel, status in results.items():
            if rel not in seen:
                f.write(f"{status} {rel}\n")


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sync_test_lists(
    conf_dir: str, discovered: list[str]
) -> tuple[list[str], dict[str, str]]:
    """Synchronize all_tests.txt and tests_result.txt"""
    all_tests_path = os.path.join(conf_dir, ALL_TESTS_FILE)
    results_path = os.path.join(conf_dir, TESTS_RESULT_FILE)

    existing_all = load_all_tests_file(all_tests_path)
    results = load_tests_result_file(results_path)

    final_tests = discovered[:]

    # Remove deleted tests
    for k in list(results.keys()):
        if k not in final_tests:
            del results[k]

    # Add new tests as not tested
    for t in final_tests:
        if t not in results:
            results[t] = STATUS_NOT_TESTED

    save_all_tests_file(all_tests_path, final_tests)
    save_tests_result_file(results_path, results, final_tests)

    return final_tests, results


def _parse_cuda_visible_devices(cuda_env: str) -> list[int]:
    """
    Parse CUDA_VISIBLE_DEVICES environment variable and return list of GPU IDs.

    Args:
        cuda_env: Value of CUDA_VISIBLE_DEVICES environment variable

    Returns:
        List of GPU device IDs
    """
    device_list = []
    for part in cuda_env.split(","):
        part = part.strip()
        if "-" in part:  # Handle range like "8-15"
            start, end = map(int, part.split("-"))
            device_list.extend(range(start, end + 1))
        else:
            device_list.append(int(part))
    return device_list


def detect_gpu(devices: list[int] | None = None) -> tuple[list[int], int]:
    """
    Detect available GPUs or use provided device list

    Args:
        devices: Optional list of GPU device IDs to use

    Returns:
        Tuple of (device_list, num_gpus)
        - device_list: List of GPU device IDs to use
        - num_gpus: Number of GPUs to use
    """
    # If devices are provided, use them directly
    if devices is not None:
        print(f"Using provided GPU device IDs: {devices}")
        return devices, len(devices)

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if cuda_env is not None:
        cuda_env = cuda_env.strip()
        if cuda_env in ("", "-1"):
            return [], 0

        try:
            # Parse GPU IDs in CUDA_VISIBLE_DEVICES
            device_list = _parse_cuda_visible_devices(cuda_env)
            return device_list, len(device_list)
        except Exception:
            pass

    # If CUDA_VISIBLE_DEVICES is not set, try to detect GPUs automatically
    # First try NVIDIA GPUs with nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], stderr=subprocess.DEVNULL, text=True
        )
        lines = [l for l in out.splitlines() if l.strip()]
        device_count = len(lines)
        if device_count > 0:
            print(f"Detected {device_count} NVIDIA GPUs using nvidia-smi")
            return list(range(device_count)), device_count
    except Exception:
        pass

    # If nvidia-smi failed, try Iluvatar GPUs with ixsmi
    try:
        out = subprocess.check_output(
            ["ixsmi", "--list-gpus"], stderr=subprocess.DEVNULL, text=True
        )
        lines = [l for l in out.splitlines() if l.strip()]
        device_count = len(lines)
        if device_count > 0:
            print(f"Detected {device_count} Iluvatar GPUs using ixsmi")
            return list(range(device_count)), device_count
    except Exception:
        pass

    # If no GPUs detected, default to single GPU (index 0)
    print("No GPUs detected automatically, defaulting to single GPU (index 0)")
    print("Use --devices parameter to specify GPU devices if needed")
    return [0], 1


def run_single_test(
    base_path: str,
    rel_path: str,
    timeout_seconds: int,
    device_index: int,
    skip_float64: bool = False,
    device_list: list[int] | None = None,
) -> tuple[str, str, str]:
    """Execute a single pytest test file"""
    print(f"Starting to run test on GPU ID {device_index}: {rel_path}")

    full_path = os.path.join(base_path, rel_path)
    env = os.environ.copy()

    # Use the provided device list to map device_index to actual GPU ID
    if device_list is not None and device_index < len(device_list):
        # Map the logical device_index to the actual GPU ID from the device list
        actual_gpu_id = device_list[device_index]
        env["CUDA_VISIBLE_DEVICES"] = str(actual_gpu_id)
    else:
        # Fallback to using device_index directly as GPU ID
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)

    # Set FLAG_SKIP_FLOAT64 environment variable if requested
    if skip_float64:
        env["FLAG_SKIP_FLOAT64"] = "1"

    env["FLAGS_enable_api_kernel_fallback"] = "0"

    # Add plugin directory to PYTHONPATH if skip_float64 is enabled
    cmd = [sys.executable, "-m", "pytest", full_path, "-q"]
    if skip_float64:
        # Add the plugin directory to PYTHONPATH
        plugin_dir = script_dir()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{plugin_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = plugin_dir

        # Add the plugin to pytest command
        cmd.extend(["-p", "float64_skip_plugin"])

    try:
        proc = subprocess.run(
            cmd,
            cwd=base_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,  # Pass custom environment variables
        )
        if proc.returncode == 0:
            return STATUS_PASSED, proc.stdout, proc.stderr
        else:
            return STATUS_FAILED, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return STATUS_TIMEOUT, e.stdout or "", e.stderr or ""
    except Exception as e:
        return STATUS_FAILED, "", f"Exception running pytest: {e}"


def save_failed_log(conf_dir: str, rel_path: str, stdout, stderr):
    """Save failed or timeout logs, compatible with stdout/stderr being bytes"""
    log_dir = os.path.join(conf_dir, FAILED_LOG_DIR)
    ensure_dir(log_dir)
    base = os.path.basename(rel_path)
    log_name = f"{os.path.splitext(base)[0]}.log"
    log_path = os.path.join(log_dir, log_name)

    # Convert to string uniformly
    def to_text(x):
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        elif x is None:
            return ""
        else:
            return str(x)

    stdout_str = to_text(stdout)
    stderr_str = to_text(stderr)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(stdout_str)
        f.write("\n\n=== STDERR ===\n")
        f.write(stderr_str)


def main():
    parser = argparse.ArgumentParser(
        description="Execute pytest tests in parallel and manage results"
    )
    parser.add_argument(
        "--path",
        "-p",
        default=".",
        help="Target path to scan (default: current directory)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=TIMEOUT_SECONDS,
        help="Test timeout in seconds",
    )
    parser.add_argument(
        "--skip-float64",
        action="store_true",
        help="Skip float64 tests by setting FLAG_SKIP_FLOAT64 environment variable",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        help="Comma-separated list of GPU device IDs to use (e.g., 0,1,2,3). If not specified, will auto-detect or default to single GPU.",
    )
    args = parser.parse_args()

    base_path = os.path.abspath(args.path)
    conf_dir = script_dir()
    timeout_seconds = args.timeout
    skip_float64 = args.skip_float64

    # Parse devices argument if provided
    device_list_arg = None
    if args.devices:
        try:
            device_list_arg = _parse_cuda_visible_devices(args.devices)
            if not device_list_arg:
                print(
                    f"Warning: Invalid devices argument '{args.devices}', will auto-detect GPUs"
                )
                device_list_arg = None
        except Exception as e:
            print(
                f"Warning: Failed to parse devices argument '{args.devices}': {e}, will auto-detect GPUs"
            )
            device_list_arg = None

    print(f"Test path: {base_path}")
    print(
        f"Configuration file directory (all_tests.txt / tests_result.txt etc.): {conf_dir}"
    )
    if skip_float64:
        print("Float64 tests will be skipped (FLAG_SKIP_FLOAT64=1)")
    if device_list_arg:
        print(f"Using specified GPU devices: {device_list_arg}")

    # 1) Discover test files
    discovered = find_test_files(base_path)
    print(f"Found {len(discovered)} Python files starting with 'test_'")

    # 2) Read disabled rules
    patterns = read_disabled_patterns(conf_dir)
    if patterns:
        print("Disabled rules:")
        for p in patterns:
            print("  ", p)
    filtered = filter_disabled(discovered, patterns)
    removed = set(discovered) - set(filtered)
    if removed:
        print("The following files are disabled:")
        for r in sorted(removed):
            print("  ", r)

    # 3) Synchronize files
    final_tests, results = sync_test_lists(conf_dir, filtered)
    print(f"Final number of tests to manage: {len(final_tests)}")

    # 4) Determine tests to run
    to_run = [
        t
        for t in final_tests
        if results.get(t) in (STATUS_NOT_TESTED, STATUS_TIMEOUT)
    ]
    print(
        f"Will execute {len(to_run)} test files this time (skipping [PASSED])"
    )

    # === Parallel execution part ===
    device_list, num_gpus = detect_gpu(device_list_arg)
    if num_gpus <= 0:
        print(
            "[Warning] No GPU detected, will fall back to single-threaded execution."
        )
        num_gpus = 1
        device_list = [0]
    else:
        print(f"Using {num_gpus} GPUs, will execute tests in parallel.")
        # Print GPU device information
        print(f"Available GPU device IDs: {device_list}")
        print(
            "Note: In subprocesses, these GPUs will be remapped to indices 0 to N-1"
        )

    ensure_dir(os.path.join(conf_dir, FAILED_LOG_DIR))
    results_path = os.path.join(conf_dir, TESTS_RESULT_FILE)

    # Dynamic task scheduling: execute the next task on whichever GPU is idle
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_gpus
    ) as executor:
        future_to_gpu = {}
        tasks_iter = iter(to_run)
        active_futures = {}

        # Initialize one task per GPU
        for device_index in range(num_gpus):
            try:
                test = next(tasks_iter)
            except StopIteration:
                break
            fut = executor.submit(
                run_single_test,
                base_path,
                test,
                timeout_seconds,
                device_index,
                skip_float64,
                device_list,
            )
            active_futures[fut] = (test, device_index)

        while active_futures:
            done, _ = concurrent.futures.wait(
                active_futures.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                status, stdout, stderr = fut.result()
                test, device_index = active_futures.pop(fut)
                results[test] = status
                print(f"\n{test}: {status} (GPU ID: {device_index})")
                if status in (STATUS_FAILED):
                    save_failed_log(conf_dir, test, stdout, stderr)
                    print(
                        f"  Log saved to failed_logs/{os.path.basename(test)}.log"
                    )
                save_tests_result_file(results_path, results, final_tests)

                # This GPU is idle, get the next task
                try:
                    next_test = next(tasks_iter)
                    new_future = executor.submit(
                        run_single_test,
                        base_path,
                        next_test,
                        timeout_seconds,
                        device_index,
                        skip_float64,
                        device_list,
                    )
                    active_futures[new_future] = (next_test, device_index)
                except StopIteration:
                    pass
    # === End of parallel part ===

    print("\n[Fininsed] All tests executed.")
    print(f"Result file: {os.path.join(conf_dir, TESTS_RESULT_FILE)}")
    print(f"Failed log directory: {os.path.join(conf_dir, FAILED_LOG_DIR)}")


if __name__ == "__main__":
    main()
