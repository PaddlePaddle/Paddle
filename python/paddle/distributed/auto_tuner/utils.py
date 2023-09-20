# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import csv
import itertools
import os
import re
from typing import Tuple


def divisor(num, reverse=False):
    """Return the divisor of the given number."""
    results = set()
    i = 1
    mid = num // 2 + 1
    while i < mid:
        if num % i == 0:
            results.add(i)
            results.add(num // i)
        i += 1
    results = list(results)
    return sorted(results, reverse=reverse)


def dist_degree(mode, num_gpus, num_nodes):
    """Return the degree of different parallel modes by gpus and nodes num."""
    assert mode in ["dp", "mp", "pp", "sharding"]
    results = []
    if mode == "dp":
        results = divisor(num_gpus, reverse=False)

    elif mode == "pp":
        if num_nodes > 1:
            results = list(range(1, num_nodes + 1))
        else:
            results = divisor(num_gpus, reverse=True)

    elif mode == "mp":
        gpus_per_node = num_gpus // num_nodes
        results = divisor(gpus_per_node, reverse=True)

    elif mode == "sharding":
        results = divisor(num_gpus, reverse=True)

    return results


def default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    num_gpus = tuner_cfg["num_gpus"]
    num_nodes = tuner_cfg["nodes"]
    assert num_gpus > 0

    if tuner_cfg.get("dp_degree", None) == "auto":
        candidates["dp_degree"] = dist_degree("dp", num_gpus, num_nodes)
    elif tuner_cfg.get("dp_degree", None):
        candidates["dp_degree"] = tuner_cfg.get("dp_degree")
    else:
        candidates["dp_degree"] = [1]

    if tuner_cfg.get("mp_degree", None) == "auto":
        candidates["mp_degree"] = dist_degree("mp", num_gpus, num_nodes)
    elif tuner_cfg.get("mp_degree", None):
        candidates["mp_degree"] = tuner_cfg.get("mp_degree")
    else:
        candidates["mp_degree"] = [1]

    if tuner_cfg.get("pp_degree", None) == "auto":
        candidates["pp_degree"] = dist_degree("pp", num_gpus, num_nodes)
    elif tuner_cfg.get("pp_degree", None):
        candidates["pp_degree"] = tuner_cfg.get("pp_degree")
    else:
        candidates["pp_degree"] = [1]

    if tuner_cfg.get("sharding_degree", None) == "auto":
        candidates["sharding_degree"] = dist_degree(
            "sharding", num_gpus, num_nodes
        )
    elif tuner_cfg.get("sharding_degree", None):
        candidates["sharding_degree"] = tuner_cfg.get("sharding_degree")
    else:
        candidates["sharding_degree"] = [1]

    if tuner_cfg.get("sharding_stage", None) == "auto":
        candidates["sharding_stage"] = [1, 2, 3]
    elif tuner_cfg.get("sharding_stage", None):
        candidates["sharding_stage"] = tuner_cfg.get("sharding_stage")
    else:
        candidates["sharding_stage"] = [None]

    if tuner_cfg.get("use_recompute", None) == "auto":
        candidates["use_recompute"] = [False, True]
    elif tuner_cfg.get("use_recompute", None):
        candidates["use_recompute"] = tuner_cfg.get("use_recompute")
    else:
        candidates["use_recompute"] = [None]

    if tuner_cfg.get("recompute_granularity", None) == "auto":
        candidates["recompute_granularity"] = ["full_attn", "full"]
    elif tuner_cfg.get("recompute_granularity", None):
        candidates["recompute_granularity"] = tuner_cfg.get(
            "recompute_granularity"
        )
    else:
        candidates["recompute_granularity"] = [None]

    if tuner_cfg.get("micro_batch_size", None) == "auto":
        candidates["micro_batch_size"] = list(
            range(tuner_cfg["model_cfg"]["global_batch_size"], 0, -1)
        )
    elif tuner_cfg.get("micro_batch_size", None):
        candidates["micro_batch_size"] = tuner_cfg.get("micro_batch_size")
    else:
        candidates["micro_batch_size"] = [None]

    return candidates


def search_all(tuner_cfg):
    """Permutate the candidates of all hyper params."""
    candidates = tuner_cfg["candidates"]
    # Order: dp -> sharding -> mbs -> pp -> mp  -> recompute
    dp_degree_candidates = candidates["dp_degree"]
    mp_degree_candidates = candidates["mp_degree"]
    pp_degree_candidates = candidates["pp_degree"]
    mbs_candidates = candidates["micro_batch_size"]
    sharding_stage_candidates = candidates["sharding_stage"]
    sharding_degree_candidates = candidates["sharding_degree"]
    use_recompute_candidates = candidates["use_recompute"]
    recompute_granularity_candidates = candidates["recompute_granularity"]
    all_cfgs = list(
        itertools.product(
            dp_degree_candidates,
            sharding_degree_candidates,
            sharding_stage_candidates,
            mbs_candidates,
            pp_degree_candidates,
            mp_degree_candidates,
            use_recompute_candidates,
            recompute_granularity_candidates,
        )
    )
    mapping = {
        0: "dp_degree",
        1: "sharding_degree",
        2: "sharding_stage",
        3: "micro_batch_size",
        4: "pp_degree",
        5: "mp_degree",
        6: "use_recompute",
        7: "recompute_granularity",
    }
    new_all_cfgs = []
    for cfg in all_cfgs:
        new_cfg = {}
        for idx, val in enumerate(cfg):
            new_cfg[mapping[idx]] = val
        new_all_cfgs.append(new_cfg)
    return new_all_cfgs


def gen_new_args(raw_args, cfg, tuner_cfg):
    """Generate new script args."""
    assert "run_cmd" in tuner_cfg
    cmd = copy.deepcopy(tuner_cfg["run_cmd"])
    res_args = copy.deepcopy(raw_args)
    if "dp_degree" in cmd and "dp_degree" in cfg:
        if "--" in cmd["dp_degree"][0]:
            cmd["dp_degree"][1] = cmd["dp_degree"][1] + str(cfg["dp_degree"])
            res_args.extend(cmd["dp_degree"])
        else:
            cmd["dp_degree"][1] = (
                cmd["dp_degree"][1] + "=" + str(cfg["dp_degree"])
            )
            res_args.extend(cmd["dp_degree"])

    if "mp_degree" in cmd and "mp_degree" in cfg:
        if "--" in cmd["mp_degree"][0]:
            cmd["mp_degree"][1] = cmd["mp_degree"][1] + str(cfg["mp_degree"])
            res_args.extend(cmd["mp_degree"])
        else:
            cmd["mp_degree"][1] = (
                cmd["mp_degree"][1] + "=" + str(cfg["mp_degree"])
            )
            res_args.extend(cmd["mp_degree"])

    if "pp_degree" in cmd and "pp_degree" in cfg:
        if "--" in cmd["pp_degree"][0]:
            cmd["pp_degree"][1] = cmd["pp_degree"][1] + str(cfg["pp_degree"])
            res_args.extend(cmd["pp_degree"])
        else:
            cmd["pp_degree"][1] = (
                cmd["pp_degree"][1] + "=" + str(cfg["pp_degree"])
            )
            res_args.extend(cmd["pp_degree"])

    if "micro_batch_size" in cmd and "micro_batch_size" in cfg:
        if "--" in cmd["micro_batch_size"][0]:
            cmd["micro_batch_size"][1] = cmd["micro_batch_size"][1] + str(
                cfg["micro_batch_size"]
            )
            res_args.extend(cmd["micro_batch_size"])
        else:
            cmd["micro_batch_size"][1] = (
                cmd["micro_batch_size"][1] + "=" + str(cfg["micro_batch_size"])
            )
            res_args.extend(cmd["micro_batch_size"])

    if "sharding_degree" in cmd and "sharding_degree" in cfg:
        if "--" in cmd["sharding_degree"][0]:
            cmd["sharding_degree"][1] = cmd["sharding_degree"][1] + str(
                cfg["sharding_degree"]
            )
            res_args.extend(cmd["sharding_degree"])
        else:
            cmd["sharding_degree"][1] = (
                cmd["sharding_degree"][1] + "=" + str(cfg["sharding_degree"])
            )
            res_args.extend(cmd["sharding_degree"])

    if "sharding_stage" in cmd and "sharding_stage" in cfg:
        if "--" in cmd["sharding_stage"][0]:
            cmd["sharding_stage"][1] = cmd["sharding_stage"][1] + str(
                cfg["sharding_stage"]
            )
            res_args.extend(cmd["sharding_stage"])
        else:
            cmd["sharding_stage"][1] = (
                cmd["sharding_stage"][1] + "=" + str(cfg["sharding_stage"])
            )
            res_args.extend(cmd["sharding_stage"])

    if "use_recompute" in cmd and "use_recompute" in cfg:
        if "--" in cmd["use_recompute"][0]:
            cmd["use_recompute"][1] = cmd["use_recompute"][1] + str(
                cfg["use_recompute"]
            )
            res_args.extend(cmd["use_recompute"])
        else:
            cmd["use_recompute"][1] = (
                cmd["use_recompute"][1] + "=" + str(cfg["use_recompute"])
            )
            res_args.extend(cmd["use_recompute"])

    if "recompute_granularity" in cmd and "recompute_granularity" in cfg:
        if "--" in cmd["recompute_granularity"][0]:
            cmd["recompute_granularity"][1] = cmd["recompute_granularity"][
                1
            ] + str(cfg["recompute_granularity"])
            res_args.extend(cmd["recompute_granularity"])
        else:
            cmd["recompute_granularity"][1] = (
                cmd["recompute_granularity"][1]
                + "="
                + str(cfg["recompute_granularity"])
            )
            res_args.extend(cmd["recompute_granularity"])

    if "local_batch_size" in cmd:
        local_batch_size = (
            tuner_cfg["model_cfg"]["global_batch_size"]
            // cfg["sharding_degree"]
            // cfg["dp_degree"]
        )
        if "--" in cmd["local_batch_size"][0]:
            cmd["local_batch_size"][1] = cmd["local_batch_size"][1] + str(
                local_batch_size
            )
            res_args.extend(cmd["local_batch_size"])
        else:
            cmd["local_batch_size"][1] = (
                cmd["local_batch_size"][1] + "=" + str(local_batch_size)
            )
            res_args.extend(cmd["local_batch_size"])

    if "gradient_accumulation_steps" in cmd:
        if "--" in cmd["gradient_accumulation_steps"][0]:
            try:
                gradient_accumulation_steps = (
                    tuner_cfg["model_cfg"]["global_batch_size"]
                    // cfg["sharding_degree"]
                    // cfg["dp_degree"]
                    // cfg["micro_batch_size"]
                )
                cmd["gradient_accumulation_steps"][1] = cmd[
                    "gradient_accumulation_steps"
                ][1] + str(gradient_accumulation_steps)
                res_args.extend(cmd["gradient_accumulation_steps"])
            except:
                pass
        else:
            try:
                gradient_accumulation_steps = (
                    tuner_cfg["model_cfg"]["global_batch_size"]
                    // cfg["sharding_degree"]
                    // cfg["dp_degree"]
                    // cfg["micro_batch_size"]
                )
                cmd["gradient_accumulation_steps"][1] = (
                    cmd["gradient_accumulation_steps"][1]
                    + "="
                    + str(gradient_accumulation_steps)
                )
                res_args.extend(cmd["gradient_accumulation_steps"])
            except:
                pass

    return res_args


def read_metric_log(
    path, file="workerlog.0", target_metric='step/s'
) -> Tuple[float, int]:
    """For extracting metric from log file."""
    """
    return:
        metric: average metric of last 10 steps
        err_code:
            00: no error
            01: no metric
            10: out of memory
    """
    err_code = 0
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return (0.0, 1)
    with open(target_file, "r") as f:
        # read file
        re_metric_pattern = (
            target_metric + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + target_metric
        )
        re_out_of_memory_pattern = r"Out of memory"
        out_of_memory_flag = 0
        metric_list = []
        lines = f.readlines()
        for line in lines:
            metric = re.findall(re_metric_pattern, line)
            out_of_memory = re.findall(
                re_out_of_memory_pattern, line, re.IGNORECASE
            )
            if metric:
                metric_list.append(float(metric[0][0]))
            if out_of_memory:
                out_of_memory_flag = 1

        if out_of_memory_flag:
            metric_ave = 0.0
            err_code = err_code | (out_of_memory_flag << 1)
        if not metric_list:
            metric_ave = 0.0
            err_code = err_code | 1
        elif len(metric_list) < 10:
            metric_ave = metric_list[-1]
        elif len(metric_list) < 20:
            metric_ave = sum(metric_list[9:]) / (len(metric_list[9:]))
        else:
            metric_ave = sum(metric_list[-10:]) / 10
        # round to 5 decimal places
        metric_ave = round(metric_ave, 5)
    res = metric_ave, err_code
    return res


def read_memory_log(path, file) -> Tuple[float, bool]:
    log_path = os.path.join(path, file)
    if not os.path.exists(log_path):
        return (0.0, True)
    memory_used = []
    utilization_gpu = []
    indexs = []

    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        flag = False
        # skip headers
        while not flag:
            # show the first line of reader
            row = next(reader)
            if len(row) == 6 and 'memory_used' in row:
                flag = True
        for row in reader:
            # If row length is 6 then it's a utilization data row
            # skip header
            if len(row) == 6:
                index, util_gpu, _, mem_used, _, _ = row
                indexs.append(int(index))
                memory_used.append(int(mem_used))
                utilization_gpu.append(int(util_gpu))
    return max(memory_used), False


def read_log(
    path,
    metric_file="workerlog.0",
    target_metric='step/s',
    memory_file="0.gpu.log",
) -> Tuple[float, float, int]:
    """
    extract metric and max memory usage from log file
    return:
        metric: average metric of last 10 steps
        memory: max memory used
        err_code: 00: no error, 01: no metric, 10: out of memory, 100: no memory log
    """
    err_code = 0
    # check out of memory
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith("workerlog"):
                continue
            metric, metric_flag = read_metric_log(path, file, target_metric)
            if metric_flag:
                err_code = (metric_flag & 2) | err_code

    # read metric
    res_metric, metric_flag = read_metric_log(path, metric_file, target_metric)
    err_code = metric_flag | err_code
    # check max memory usage
    try:
        res_memory, memory_flag = read_memory_log(path, memory_file)
        err_code = (memory_flag << 2) | err_code
    except:
        res_memory = 0.0
        err_code = (1 << 2) | err_code
    return res_metric, res_memory, err_code


def three_mul_combinations(target):
    """Return the combinations of three numbers which product is target."""
    results = []
    for i in range(1, target // 3 + 1):
        if target % i == 0:
            for j in range(i, target // 2 + 1):
                if (target // i) % j == 0:
                    results.append((i, j, target // i // j))
    return results


def gbs_dp_mp_pp_candidates(tuner_cfg, num_gpus, num_nodes):
    """Return middle candidates of dp, mp, pp"""

    start = round(num_gpus ** (1 / 3))

    # find factors that can be evenly distributed
    for i in range(start, 0, -1):
        if num_gpus % i == 0:
            remaining = num_gpus // i
            # find the square root as a factor for the remaining part
            j = round(remaining**0.5)
            while remaining % j != 0:
                j -= 1
            return i, j, remaining // j

    raise ValueError("Cannot distribute GPUs equally")


def gbs_default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    num_gpus = tuner_cfg["num_gpus"]
    num_nodes = tuner_cfg["nodes"]
    assert num_gpus > 0
    global_batch_size = tuner_cfg.get("model_cfg", {}).get(
        "global_batch_size", "auto"
    )
    if global_batch_size == "auto":
        dp_candidate, mp_candidate, pp_candidate = gbs_dp_mp_pp_candidates(
            tuner_cfg, num_gpus, num_nodes
        )
        sharding_dgree_candidate = dp_candidate
        candidates["dp_degree"] = [1]
        candidates["mp_degree"] = [mp_candidate]
        candidates["pp_degree"] = [pp_candidate]
        candidates["sharding_degree"] = [sharding_dgree_candidate]
        candidates["sharding_stage"] = [1]
        candidates["use_recompute"] = [False]
        candidates["recompute_granularity"] = [None]
        candidates["micro_batch_size"] = [2**i for i in range(0, 10)]
        candidates["global_batch_size"] = [
            pp_candidate * dp_candidate * e
            for e in candidates["micro_batch_size"]
        ]
    return candidates


def gbs_search_all(tuner_cfg):
    """Permutate the candidates of all hyper params."""
    candidates = tuner_cfg["candidates"]
    # Order: dp -> mp -> pp -> mbs -> sharding-> recompute
    dp_degree_candidates = candidates["dp_degree"]
    mp_degree_candidates = candidates["mp_degree"]
    pp_degree_candidates = candidates["pp_degree"]
    mbs_candidates = candidates["micro_batch_size"]
    sharding_stage_candidates = candidates["sharding_stage"]
    sharding_degree_candidates = candidates["sharding_degree"]
    use_recompute_candidates = candidates["use_recompute"]
    recompute_granularity_candidates = candidates["recompute_granularity"]
    # gbs_candidates = candidates["global_batch_size"]
    all_cfgs = list(
        itertools.product(
            dp_degree_candidates,
            mp_degree_candidates,
            pp_degree_candidates,
            mbs_candidates,
            sharding_degree_candidates,
            sharding_stage_candidates,
            use_recompute_candidates,
            recompute_granularity_candidates,
            # gbs_candidates,
        )
    )
    mapping = {
        0: "dp_degree",
        1: "mp_degree",
        2: "pp_degree",
        3: "micro_batch_size",
        5: "sharding_stage",
        4: "sharding_degree",
        6: "use_recompute",
        7: "recompute_granularity",
        # 8: "global_batch_size",
    }
    new_all_cfgs = []
    for cfg in all_cfgs:
        new_cfg = {}
        for idx, val in enumerate(cfg):
            new_cfg[mapping[idx]] = val
        new_cfg["global_batch_size"] = (
            new_cfg["pp_degree"]
            * new_cfg["dp_degree"]
            * new_cfg["micro_batch_size"]
        )
        new_all_cfgs.append(new_cfg)
    return new_all_cfgs
