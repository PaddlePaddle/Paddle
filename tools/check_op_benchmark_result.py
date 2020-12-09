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

import os
import json
import logging
import argparse


def check_path_exists(path):
    """Assert whether file/directory exists.
    """
    assert os.path.exists(path), "%s does not exist." % path


def parse_log_file(log_file):
    """Load one case result from log file.
    """
    check_path_exists(log_file)

    result = None
    with open(log_file) as f:
        for line in f.read().strip().split('\n')[::-1]:
            try:
                result = json.loads(line)
                return result
            except ValueError:
                pass  # do nothing

    if result is None:
        logging.warning("Parse %s fail!" % log_file)

    return result


def load_benchmark_result_from_logs_dir(logs_dir):
    """Load benchmark result from logs directory.
    """
    check_path_exists(logs_dir)

    log_file_path = lambda log_file: os.path.join(logs_dir, log_file)
    result_lambda = lambda log_file: (log_file, parse_log_file(log_file_path(log_file)))

    return dict(map(result_lambda, os.listdir(logs_dir)))


def compare_benchmark_result(develop_result, pr_result):
    """Compare the differences between devlop and pr.
    """
    status = True
    develop_speed = develop_result.get("speed")
    pr_speed = pr_result.get("speed")

    assert type(develop_speed) == type(
        pr_speed), "The types of comparison results need to be consistent."

    if isinstance(develop_speed, dict) and isinstance(pr_speed, dict):
        pr_gpu_time = pr_speed.get("gpu_time")
        develop_gpu_time = develop_speed.get("gpu_time")
        gpu_time_diff = (pr_gpu_time - develop_gpu_time) / develop_gpu_time

        pr_total_time = pr_speed.get("total")
        develop_total_time = develop_speed.get("total")
        total_time_diff = (
            pr_total_time - develop_total_time) / develop_total_time

        if gpu_time_diff > 0.05:
            status = False

        # TODO(Avin0323): Print all info for making relu of alart.
        logging.info("------ OP: %s ------" % pr_result.get("name"))
        logging.info("GPU time change: %.5f%% (develop: %.7f -> PR: %.7f)" %
                     (gpu_time_diff * 100, develop_gpu_time, pr_gpu_time))
        logging.info("Total time change: %.5f%% (develop: %.7f -> PR: %.7f)" %
                     (total_time_diff * 100, develop_total_time, pr_total_time))
        logging.info("backward: %s" % pr_result.get("backward"))
        logging.info("parameters:")
        for line in pr_result.get("parameters").strip().split("\n"):
            logging.info("\t%s" % line)
    else:
        if not pr_result.get("consistent"):
            status = False
            logging.info("------ OP: %s ------" % pr_result.get("name"))
            logging.info("Accaury diff: %s" % pr_result.get("diff"))
            logging.info("backward: %s" % pr_result.get("backward"))
            logging.info("parameters:")
            for line in pr_result.get("parameters").strip().split("\n"):
                logging.info("\t%s" % line)

    return status


if __name__ == "__main__":
    """Load result from log directories and compare the differences.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s:%(lineno)d] [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--develop_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of develop branch.")
    parser.add_argument(
        "--pr_logs_dir",
        type=str,
        required=True,
        help="Specify the benchmark result directory of PR branch.")
    args = parser.parse_args()

    exit_code = 0

    develop_result_dict = load_benchmark_result_from_logs_dir(
        args.develop_logs_dir)

    check_path_exists(args.pr_logs_dir)
    for log_file in os.listdir(args.pr_logs_dir):
        develop_result = develop_result_dict.get(log_file)
        pr_result = parse_log_file(os.path.join(args.pr_logs_dir, log_file))
        if develop_result is None or pr_result is None:
            continue
        if not compare_benchmark_result(develop_result, pr_result):
            exit_code = 8

    exit(exit_code)
