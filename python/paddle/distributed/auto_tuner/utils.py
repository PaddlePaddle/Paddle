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
    if num == 1:
        return [num]
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


def dist_degree(mode, num_gpus, num_nodes, tuner_cfg=None):
    """Return the degree of different parallel modes by gpus and nodes num."""
    assert mode in ["dp", "mp", "pp", "sharding", "mbs", "vpp"]
    results = []
    prune_results = []
    if mode == "dp":
        results = divisor(num_gpus, reverse=False)

    elif mode == "pp":
        if num_nodes > 1 and tuner_cfg.get("enable_pp_prune", True):
            results = list(range(num_nodes + 1, 0, -1))
        else:
            results = divisor(num_gpus, reverse=True)
        for pp_degree in results:
            prune_flag = False
            num_layers = tuner_cfg["model_cfg"].get("num_layers", None)

            if num_layers:
                if num_layers % pp_degree != 0:
                    prune_flag = True

            if not prune_flag:
                prune_results.append(pp_degree)
        results = prune_results

    elif mode == "mp":
        if tuner_cfg.get("enable_mp_prune", True):
            gpus_per_node = num_gpus // num_nodes
            results = divisor(gpus_per_node, reverse=True)
        else:
            results = divisor(num_gpus, reverse=True)
        for mp_degree in results:
            prune_flag = False
            hidden_size = tuner_cfg["model_cfg"].get("hidden_size", None)
            vocab_size = tuner_cfg["model_cfg"].get("vocab_size", None)
            num_attention_heads = tuner_cfg["model_cfg"].get(
                "num_attention_heads", None
            )
            seq_length = tuner_cfg["model_cfg"].get("seq_length", None)
            use_sequence_paralel = tuner_cfg.get("use_sequence_paralel", False)

            if hidden_size and hidden_size % mp_degree != 0:
                prune_flag = True

            if vocab_size and vocab_size % mp_degree != 0:
                prune_flag = True

            if num_attention_heads and num_attention_heads % mp_degree != 0:
                prune_flag = True

            if (
                seq_length
                and seq_length % mp_degree != 0
                and use_sequence_paralel
            ):
                prune_flag = True

            if not prune_flag:
                prune_results.append(mp_degree)
        results = prune_results

    elif mode == "sharding":
        results = divisor(num_gpus, reverse=True)

    elif mode == "mbs":
        results = divisor(tuner_cfg["model_cfg"]["global_batch_size"])

    elif mode == "vpp":
        results = divisor(tuner_cfg["model_cfg"]["num_layers"], reverse=True)

    return results


def default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    num_gpus = tuner_cfg["num_gpus"]
    num_nodes = tuner_cfg["nodes"]
    assert num_gpus > 0

    if tuner_cfg.get("dp_degree", None) == "auto":
        candidates["dp_degree"] = dist_degree(
            "dp", num_gpus, num_nodes, tuner_cfg
        )
    elif tuner_cfg.get("dp_degree", None):
        candidates["dp_degree"] = tuner_cfg.get("dp_degree")
    else:
        candidates["dp_degree"] = [1]

    if tuner_cfg.get("mp_degree", None) == "auto":
        candidates["mp_degree"] = dist_degree(
            "mp", num_gpus, num_nodes, tuner_cfg
        )
    elif tuner_cfg.get("mp_degree", None):
        candidates["mp_degree"] = tuner_cfg.get("mp_degree")
    else:
        candidates["mp_degree"] = [1]

    if tuner_cfg.get("pp_degree", None) == "auto":
        candidates["pp_degree"] = dist_degree(
            "pp", num_gpus, num_nodes, tuner_cfg
        )
    elif tuner_cfg.get("pp_degree", None):
        candidates["pp_degree"] = tuner_cfg.get("pp_degree")
    else:
        candidates["pp_degree"] = [1]

    if tuner_cfg.get("vpp_degree", None) == "auto":
        candidates["vpp_degree"] = dist_degree(
            "vpp", num_gpus, num_nodes, tuner_cfg
        )
    elif tuner_cfg.get("vpp_degree", None):
        candidates["vpp_degree"] = tuner_cfg.get("vpp_degree")
    else:
        candidates["vpp_degree"] = [1]

    if tuner_cfg.get("sharding_degree", None) == "auto":
        candidates["sharding_degree"] = dist_degree(
            "sharding", num_gpus, num_nodes, tuner_cfg
        )
    elif tuner_cfg.get("sharding_degree", None):
        candidates["sharding_degree"] = tuner_cfg.get("sharding_degree")
    else:
        candidates["sharding_degree"] = [1]

    if tuner_cfg.get("sharding_stage", None) == "auto":
        candidates["sharding_stage"] = [3, 2, 1]
    elif tuner_cfg.get("sharding_stage", None):
        candidates["sharding_stage"] = tuner_cfg.get("sharding_stage")
    else:
        candidates["sharding_stage"] = [None]

    if tuner_cfg.get("use_recompute", None) == "auto":
        candidates["use_recompute"] = [True, False]
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
        candidates["micro_batch_size"] = dist_degree(
            "mbs", num_gpus, num_nodes, tuner_cfg
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
    vpp_degree_candidates = candidates["vpp_degree"]
    mbs_candidates = candidates["micro_batch_size"]
    sharding_stage_candidates = candidates["sharding_stage"]
    sharding_degree_candidates = candidates["sharding_degree"]
    use_recompute_candidates = candidates["use_recompute"]
    recompute_granularity_candidates = candidates["recompute_granularity"]

    num_gpus = tuner_cfg["num_gpus"]
    valid_degrees = []

    for mp_degree in mp_degree_candidates:
        degrees = []
        if num_gpus % mp_degree != 0:
            continue
        degrees.append(mp_degree)
        sharding_res = num_gpus // mp_degree

        for sharding_degree in sharding_degree_candidates:
            if sharding_res % sharding_degree != 0:
                continue
            degrees.append(sharding_degree)
            pp_res = sharding_res // sharding_degree

            for pp_degree in pp_degree_candidates:
                if pp_res % pp_degree != 0:
                    continue
                degrees.append(pp_degree)
                dp_res = pp_res // pp_degree

                for dp_degree in dp_degree_candidates:
                    if dp_res != dp_degree:
                        continue
                    degrees.append(dp_degree)
                    assert len(degrees) == 4
                    valid_degrees.append(copy.deepcopy(degrees))
                    degrees.pop()
                degrees.pop()
            degrees.pop()

    other_dim_cfgs = list(
        itertools.product(
            sharding_stage_candidates,
            mbs_candidates,
            vpp_degree_candidates,
            use_recompute_candidates,
            recompute_granularity_candidates,
        )
    )

    all_cfgs = []
    for valid_degree in valid_degrees:
        for other_dim_cfg in other_dim_cfgs:
            mp_degree, sharding_degree, pp_degree, dp_degree = valid_degree
            (
                sharding_stage,
                mbs,
                vpp,
                use_recompute,
                recompute_granularity,
            ) = list(other_dim_cfg)
            if (
                tuner_cfg["model_cfg"]["global_batch_size"]
                % (mbs * sharding_degree * dp_degree)
                != 0
            ):
                continue
            if tuner_cfg["model_cfg"]["num_layers"] % (pp_degree * vpp) != 0:
                continue
            cfg = list(valid_degree) + list(other_dim_cfg)
            all_cfgs.append(cfg)

    mapping = {
        0: "mp_degree",
        1: "sharding_degree",
        2: "pp_degree",
        3: "dp_degree",
        4: "sharding_stage",
        5: "micro_batch_size",
        6: "vpp_degree",
        7: "use_recompute",
        8: "recompute_granularity",
    }
    new_all_cfgs = []
    for cfg in all_cfgs:
        new_cfg = {}
        for idx, val in enumerate(cfg):
            new_cfg[mapping[idx]] = val
        new_all_cfgs.append(new_cfg)
    return new_all_cfgs


def gen_new_args(raw_args, cfg, tuner_cfg, run_best=False):
    """Generate new script args."""

    def _gen_new_arg(arg, cmd, cfg, res_args, tuner_cfg):
        if arg in cmd and arg in cfg:
            if "--" in cmd[arg][0]:
                cmd[arg][1] = cmd[arg][1] + str(cfg[arg])
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                cmd[arg][1] = cmd[arg][1] + "=" + str(cfg[arg])
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cfg[arg]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cfg[arg]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))
        elif arg == "local_batch_size" and arg in cmd:
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
            elif "-o" in cmd["local_batch_size"][0]:
                cmd["local_batch_size"][1] = (
                    cmd["local_batch_size"][1] + "=" + str(local_batch_size)
                )
                res_args.extend(cmd["local_batch_size"])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = local_batch_size
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = local_batch_size
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

        elif arg == "gradient_accumulation_steps" and arg in cmd:
            try:
                gradient_accumulation_steps = (
                    tuner_cfg["model_cfg"]["global_batch_size"]
                    // cfg["sharding_degree"]
                    // cfg["dp_degree"]
                    // cfg["micro_batch_size"]
                )
            except:
                return
            if "--" in cmd["gradient_accumulation_steps"][0]:
                cmd["gradient_accumulation_steps"][1] = cmd[
                    "gradient_accumulation_steps"
                ][1] + str(gradient_accumulation_steps)
                res_args.extend(cmd["gradient_accumulation_steps"])

            elif "-o" in cmd["gradient_accumulation_steps"][0]:
                cmd["gradient_accumulation_steps"][1] = (
                    cmd["gradient_accumulation_steps"][1]
                    + "="
                    + str(gradient_accumulation_steps)
                )
                res_args.extend(cmd["gradient_accumulation_steps"])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = gradient_accumulation_steps
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = gradient_accumulation_steps
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

    assert "run_cmd" in tuner_cfg
    cmd = copy.deepcopy(tuner_cfg["run_cmd"])
    res_args = copy.deepcopy(raw_args)

    _gen_new_arg("dp_degree", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("mp_degree", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("pp_degree", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("vpp_degree", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("micro_batch_size", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("sharding_degree", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("sharding_stage", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("use_recompute", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("recompute_granularity", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("local_batch_size", cmd, cfg, res_args, tuner_cfg)
    _gen_new_arg("gradient_accumulation_steps", cmd, cfg, res_args, tuner_cfg)

    if tuner_cfg["run_cmd"].get("search_stage", None) and not run_best:
        cmd = copy.deepcopy(tuner_cfg["run_cmd"]["search_stage"])
        for arg in cmd:
            if "--" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cmd[arg][2]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

    if tuner_cfg["run_cmd"].get("run_best_stage", None) and run_best:
        cmd = copy.deepcopy(tuner_cfg["run_cmd"]["run_best_stage"])
        for arg in cmd:
            if "--" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif "-o" in cmd[arg][0]:
                res_args.extend(cmd[arg])
            elif ".json" in cmd[arg][0]:
                import json

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cmd[arg][2]
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                for key in keys[: len(keys) - 1]:
                    cmd_cfg = cmd_cfg[key]
                cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

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
                value = None
                for item in metric[0]:
                    try:
                        value = float(item)
                        metric_list.append(value)
                        break
                    except:
                        continue
                assert value is not None

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
