# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import json
import logging
import os
import sys


def check_path_exists(path):
    """Assert whether file/directory exists."""
    assert os.path.exists(path), f"{path} does not exist."


def parse_case_name(log_file_name):
    """Parse case name."""
    case_id, case_info = log_file_name.split("-")
    direction = case_info.split(".")[0].split("_")[-1]

    return f"{case_id} ({direction})"


def parse_log_file(log_file):
    """Load one case result from log file."""
    check_path_exists(log_file)

    result = None
    with open(log_file) as f:
        for line in f.read().strip().split('\n')[::-1]:
            try:
                result = json.loads(line)
                if result.get("disabled", False):
                    return None
                return result
            except ValueError:
                pass  # do nothing

    if result is None:
        logging.warning(f"Parse {log_file} fail!")

    return result


def load_benchmark_result_from_logs_dir(logs_dir):
    """Load benchmark result from logs directory."""
    check_path_exists(logs_dir)

    log_file_path = lambda log_file: os.path.join(logs_dir, log_file)
    result_lambda = lambda log_file: (
        log_file,
        parse_log_file(log_file_path(log_file)),
    )

    return dict(map(result_lambda, os.listdir(logs_dir)))


def check_speed_result(case_name, develop_data, pr_data, pr_result):
    """Check speed differences between develop and pr."""
    pr_gpu_time = pr_data.get("gpu_time")
    develop_gpu_time = develop_data.get("gpu_time")
    if develop_gpu_time != 0.0:
        gpu_time_diff = (pr_gpu_time - develop_gpu_time) / develop_gpu_time
        gpu_time_diff_str = f"{gpu_time_diff * 100:.5f}"
    else:
        gpu_time_diff = 0
        gpu_time_diff_str = ""

    pr_total_time = pr_data.get("total")
    develop_total_time = develop_data.get("total")
    total_time_diff = (pr_total_time - develop_total_time) / develop_total_time

    logging.info(f"------ OP: {case_name} ------")
    logging.info(
        f"GPU time change: {gpu_time_diff_str} (develop: {develop_gpu_time:.7f} -> PR: {pr_gpu_time:.7f})"
    )
    logging.info(
        f"Total time change: {total_time_diff * 100:.5f}% (develop: {develop_total_time:.7f} -> PR: {pr_total_time:.7f})"
    )
    logging.info("backward: {}".format(pr_result.get("backward")))
    logging.info("parameters:")
    for line in pr_result.get("parameters").strip().split("\n"):
        logging.info(f"\t{line}")

    return gpu_time_diff > 0.05


def check_accuracy_result(case_name, pr_result):
    """Check accuracy result."""
    logging.info(f"------ OP: {case_name} ------")
    logging.info("Accuracy diff: {}".format(pr_result.get("diff")))
    logging.info("backward: {}".format(pr_result.get("backward")))
    logging.info("parameters:")
    for line in pr_result.get("parameters").strip().split("\n"):
        logging.info(f"\t{line}")

    return not pr_result.get("consistent")


def compare_benchmark_result(
    case_name, develop_result, pr_result, check_results
):
    """Compare the differences between develop and pr."""
    develop_speed = develop_result.get("speed")
    pr_speed = pr_result.get("speed")

    assert type(develop_speed) == type(
        pr_speed
    ), "The types of comparison results need to be consistent."

    if isinstance(develop_speed, dict) and isinstance(pr_speed, dict):
        if check_speed_result(case_name, develop_speed, pr_speed, pr_result):
            check_results["speed"].append(case_name)
    else:
        if check_accuracy_result(case_name, pr_result):
            check_results["accuracy"].append(case_name)


def update_api_info_file(fail_case_list, api_info_file):
    """Update api info file to auto retry benchmark test."""
    check_path_exists(api_info_file)

    # set of case names for performance check failures
    parse_case_id_f = lambda x: x.split()[0].rsplit('_', 1)
    fail_case_dict = dict(map(parse_case_id_f, fail_case_list))

    # list of api infos for performance check failures
    api_info_list = []
    with open(api_info_file) as f:
        for line in f:
            line_list = line.split(',')
            case = line_list[0].split(':')[0]
            if case in fail_case_dict:
                line_list[0] = f"{case}:{fail_case_dict[case]}"
                api_info_list.append(','.join(line_list))

    # update api info file
    with open(api_info_file, 'w') as f:
        for api_info_line in api_info_list:
            f.write(api_info_line)


def summary_results(check_results, api_info_file):
    """Summary results and return sys.exit code."""
    for case_name in check_results["speed"]:
        logging.error(f'Check speed result with case "{case_name}" failed.')

    for case_name in check_results["accuracy"]:
        logging.error(f'Check accuracy result with case "{case_name}" failed.')

    if len(check_results["speed"]) and api_info_file:
        update_api_info_file(check_results["speed"], api_info_file)

    if len(check_results["speed"]) or len(check_results["accuracy"]):
        return 8
    else:
        return 0


if __name__ == "__main__":
    """Load result from log directories and compare the differences."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--develop_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of develop branch.",
    )
    parser.add_argument(
        "--pr_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of PR branch.",
    )
    parser.add_argument(
        "--api_info_file",
        type=str,
        required=False,
        help="Specify the api info to run benchmark test.",
    )
    args = parser.parse_args()

    check_results = {"accuracy": [], "speed": []}

    develop_result_dict = load_benchmark_result_from_logs_dir(
        args.develop_logs_dir
    )

    check_path_exists(args.pr_logs_dir)
    pr_log_files = os.listdir(args.pr_logs_dir)
    for log_file in sorted(pr_log_files):
        develop_result = develop_result_dict.get(log_file)
        pr_result = parse_log_file(os.path.join(args.pr_logs_dir, log_file))
        if develop_result is None or pr_result is None:
            continue
        case_name = parse_case_name(log_file)
        compare_benchmark_result(
            case_name, develop_result, pr_result, check_results
        )

    sys.exit(summary_results(check_results, args.api_info_file))
