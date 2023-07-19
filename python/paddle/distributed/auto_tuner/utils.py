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


def read_log(
    path, file="workerlog.0", target_metric='step/s'
) -> Tuple[float, bool]:
    """For extracting metric from log file."""
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return (0.0, True)
    with open(target_file, "r") as f:
        # read file
        re_metric_pattern = (
            target_metric + r":* *(\d+(\.\d*)?)|(\d+(\.\d*)?) *" + target_metric
        )

        metric_list = []
        lines = f.readlines()
        for line in lines:
            metric = re.findall(re_metric_pattern, line)
            if metric:
                metric_list.append(float(metric[0][0]))
        if not metric_list:
            metric_ave = 0.0
            flag = True
        elif len(metric_list) < 10:
            metric_ave = metric_list[-1]
            flag = False
        elif len(metric_list) < 20:
            metric_ave = sum(metric_list[9:]) / (len(metric_list[9:]))
            flag = False
        else:
            metric_ave = sum(metric_list[-10:]) / 10
            flag = False
        # round to 5 decimal places
        metric_ave = round(metric_ave, 5)
    res = metric_ave, flag
    return res
