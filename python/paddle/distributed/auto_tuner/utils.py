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

from .prune import _PRUNE_FUNC

__SUPPORTED_RECOMPUTE_GRANULARITY__ = ["full", "full_attn", "core_attn"]


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


def dist_degree_with_customized_range(
    mode, num_gpus, num_nodes, customized_range, tuner_cfg=None
):
    """Return the degree of different parallel modes by gpus and nodes num with customized range."""
    dist_degree_all = dist_degree(mode, num_gpus, num_nodes, tuner_cfg)
    return [degree for degree in dist_degree_all if degree in customized_range]


def dist_degree(mode, num_gpus, num_nodes, tuner_cfg=None):
    """Return the degree of different parallel modes by gpus and nodes num."""
    assert mode in [
        "dp_degree",
        "mp_degree",
        "pp_degree",
        "sharding_degree",
        "micro_batch_size",
        "vpp_degree",
    ]
    results = []
    prune_results = []
    if mode == "dp_degree":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(num_gpus, reverse=False)
        else:
            results = divisor(num_gpus, reverse=True)

    elif mode == "pp_degree":
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

    elif mode == "mp_degree":
        if tuner_cfg.get("enable_mp_prune", True):
            gpus_per_node = num_gpus // num_nodes
            if tuner_cfg.get("schedule_mode", "memory") != "performance":
                results = divisor(gpus_per_node, reverse=True)
            else:
                results = divisor(gpus_per_node, reverse=False)
        else:
            if tuner_cfg.get("schedule_mode", "memory") != "performance":
                results = divisor(num_gpus, reverse=True)
            else:
                results = divisor(num_gpus, reverse=False)

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

    elif mode == "sharding_degree":
        results = divisor(num_gpus, reverse=True)

    elif mode == "micro_batch_size":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(
                tuner_cfg["model_cfg"]["global_batch_size"], reverse=False
            )
        else:
            results = divisor(
                tuner_cfg["model_cfg"]["global_batch_size"], reverse=True
            )

    elif mode == "vpp_degree":
        if tuner_cfg.get("schedule_mode", "memory") != "performance":
            results = divisor(
                tuner_cfg["model_cfg"]["num_layers"], reverse=False
            )
        else:
            results = divisor(
                tuner_cfg["model_cfg"]["num_layers"], reverse=True
            )

    return results


def default_candidates(tuner_cfg):
    """Return the default candidates of every hyper param which user defined auto"""
    candidates = {}
    estimated_num_gpus = None
    if (
        "search_algo" in tuner_cfg
        and "estimated_num_gpus" in tuner_cfg["search_algo"]
    ):
        estimated_num_gpus = tuner_cfg["search_algo"]["estimated_num_gpus"]
    num_gpus = (
        tuner_cfg["num_gpus"]
        if estimated_num_gpus is None
        else estimated_num_gpus
    )
    num_nodes = (
        tuner_cfg["nodes"]
        if estimated_num_gpus is None
        else estimated_num_gpus // tuner_cfg["gpus_per_node"]
    )
    assert num_gpus > 0

    for strategy in ["dp_degree", "mp_degree", "pp_degree", "sharding_degree"]:
        strategy_customized_range = _param2range(
            tuner_cfg.get(strategy, None), num_gpus, strategy
        )
        candidates[strategy] = dist_degree_with_customized_range(
            strategy, num_gpus, num_nodes, strategy_customized_range, tuner_cfg
        )

    vpp_degree_customized_range = _param2range(
        tuner_cfg.get("vpp_degree", None),
        tuner_cfg["model_cfg"]["num_layers"],
        "vpp_degree",
    )
    candidates["vpp_degree"] = dist_degree_with_customized_range(
        "vpp_degree",
        num_gpus,
        num_nodes,
        vpp_degree_customized_range,
        tuner_cfg,
    )

    mbs_customized_range = _param2range(
        tuner_cfg.get("micro_batch_size", None),
        tuner_cfg["model_cfg"]["global_batch_size"],
        "micro_batch_size",
    )
    candidates["micro_batch_size"] = dist_degree_with_customized_range(
        "micro_batch_size", num_gpus, num_nodes, mbs_customized_range, tuner_cfg
    )

    schedule_mode = tuner_cfg.get("schedule_mode", "memory")
    sharding_stage_customized_range = _param2range(
        tuner_cfg.get("sharding_stage", None), 3, "sharding_stage"
    )
    candidates["sharding_stage"] = [
        stage for stage in [3, 2, 1] if stage in sharding_stage_customized_range
    ]
    if schedule_mode != "performance":
        candidates["sharding_stage"] = sorted(
            candidates["sharding_stage"], reverse=True
        )
    else:
        candidates["sharding_stage"] = sorted(
            candidates["sharding_stage"], reverse=False
        )

    use_recompute = tuner_cfg.get("use_recompute", None)
    if isinstance(use_recompute, str) and use_recompute.lower() == "auto":
        candidates["use_recompute"] = (
            [True, False] if schedule_mode != "performance" else [False, True]
        )
        tuner_cfg["recompute_granularity"] = "auto"
    elif isinstance(use_recompute, bool):
        candidates["use_recompute"] = [use_recompute]
    # TODO: should remove this case in the future
    elif use_recompute is None:
        candidates["use_recompute"] = [None]
    else:
        raise ValueError("use_recompute supports auto/True/False")

    recompute_granularity = tuner_cfg.get("recompute_granularity", None)
    if isinstance(recompute_granularity, str):
        if recompute_granularity.lower() == "auto":
            candidates["recompute_granularity"] = (
                __SUPPORTED_RECOMPUTE_GRANULARITY__
                if schedule_mode != "performance"
                else list(reversed(__SUPPORTED_RECOMPUTE_GRANULARITY__))
            )
        elif (
            recompute_granularity.lower() in __SUPPORTED_RECOMPUTE_GRANULARITY__
        ):
            candidates["recompute_granularity"] = [
                recompute_granularity.lower()
            ]
        # TODO: should remove this case in the future
        elif recompute_granularity is None:
            candidates["recompute_granularity"] = [None]
        else:
            raise ValueError(
                f"recompute_granularity only supports auto/{'/'.join(__SUPPORTED_RECOMPUTE_GRANULARITY__)}, but got {recompute_granularity}"
            )
    else:
        raise ValueError(
            f"recompute_granularity only supports auto/{'/'.join(__SUPPORTED_RECOMPUTE_GRANULARITY__)}, but got {recompute_granularity}"
        )

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

    num_gpus = (
        tuner_cfg["num_gpus"]
        if "search_algo" not in tuner_cfg
        or "estimated_num_gpus" not in tuner_cfg["search_algo"]
        else tuner_cfg["search_algo"]["estimated_num_gpus"]
    )
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

    pruned_all_cfgs = []
    tuner_cfg["num_gpus"] = num_gpus
    for cur_cfg in new_all_cfgs:
        pruned = False
        for func in _PRUNE_FUNC:
            result = func(tuner_cfg, cur_cfg, [])
            if result:
                pruned = True
                break
        if not pruned:
            pruned_all_cfgs.append(cur_cfg)
    return pruned_all_cfgs


def _param2range(param_from_json_file, max_value, param_key):
    """Convert a param from json file to candidates range."""
    selected_range = None
    if isinstance(param_from_json_file, str):
        if "auto" in param_from_json_file.lower():
            selected_range = list(range(1, max_value + 1))
        else:
            raise ValueError(
                f"Illegal param found: {param_key}, only support auto in str type."
            )
    elif isinstance(param_from_json_file, dict):
        customized_min_value = param_from_json_file.get("min", None)
        customized_max_value = param_from_json_file.get("max", None)
        if not (customized_min_value and customized_max_value):
            raise ValueError(
                f"Illegal param found: {param_key}, min and max should be specified in dict type."
            )
        selected_range = list(
            range(customized_min_value, customized_max_value + 1)
        )
    elif isinstance(param_from_json_file, list):
        selected_range = param_from_json_file
    elif isinstance(param_from_json_file, int):
        selected_range = [param_from_json_file]
    elif param_from_json_file is None:
        selected_range = [1]
    else:
        raise ValueError(
            f"Illegal param found: {param_key}, only support str, dict, list and int type."
        )
    return selected_range


def search_by_dp_estimation(tuner_cfg):
    all_cfgs = search_all(tuner_cfg)
    estimated_num_gpus = tuner_cfg["search_algo"].get(
        "estimated_num_gpus", None
    )
    assert estimated_num_gpus is not None
    # change global_batch_size, dp_degree, sharding_degree
    new_all_cfgs = []
    for task in all_cfgs:
        task["estimated_dp_degree"] = int(
            task["dp_degree"] * task["sharding_degree"]
        )
        task["dp_degree"] = 1
        task["sharding_degree"] = 1
        task["sharding_stage"] = 1
        task["num_gpus"] = task["mp_degree"] * task["pp_degree"]
        if task["num_gpus"] >= tuner_cfg["gpus_per_node"]:
            task["nodes"] = task["num_gpus"] // tuner_cfg["gpus_per_node"]
        else:
            task["nodes"] = 1
        task["global_batch_size"] = (
            tuner_cfg["model_cfg"]["global_batch_size"]
            // task["estimated_dp_degree"]
        )
        if task not in new_all_cfgs and task["nodes"] <= tuner_cfg["nodes"]:
            new_all_cfgs.append(task)

    # expanding sharding degree to run overlap and nonoverlap to calculate overlap benefits
    if tuner_cfg["search_algo"].get("sharding_overlap", None):
        sharding_all_cfgs = []
        for task in new_all_cfgs:
            new_task = copy.deepcopy(task)
            given_num_gpus = tuner_cfg["nodes"] * tuner_cfg["gpus_per_node"]
            sharding_degree = given_num_gpus // task["num_gpus"]
            if sharding_degree > 1:
                new_task["sharding_degree"] = sharding_degree
                new_task["sharding_stage"] = 1
                new_task["estimated_dp_degree"] = None
                new_task["num_gpus"] = (
                    new_task["mp_degree"]
                    * new_task["pp_degree"]
                    * new_task["sharding_degree"]
                )
                if new_task["num_gpus"] >= tuner_cfg["gpus_per_node"]:
                    new_task["nodes"] = (
                        new_task["num_gpus"] // tuner_cfg["gpus_per_node"]
                    )
                else:
                    new_task["nodes"] = 1
                new_task["global_batch_size"] = (
                    task["global_batch_size"] * sharding_degree
                )
                new_task["sharding_overlap"] = False
                sharding_all_cfgs.append(new_task)

                overlap_new_task = copy.deepcopy(new_task)
                overlap_new_task["sharding_overlap"] = True
                sharding_all_cfgs.append(overlap_new_task)

    new_all_cfgs.extend(sharding_all_cfgs)
    return new_all_cfgs


def add_overlap_performance(cur_cfg, tuner_cfg, history_cfgs):
    """
    In single dp search scenario,
    the overlay acceleration ratio is obtained by automatically running overlap and non overlap tasks,
    and the estimated performance of the multi dp after overlap is obtained.
    """
    if cur_cfg[tuner_cfg['metric_cfg']['name']]:
        non_overlap_cfg = None
        raw_cfg = None
        for cfg in history_cfgs:
            keys = [
                "dp_degree",
                "mp_degree",
                "pp_degree",
                "vpp_degree",
                "micro_batch_size",
                "use_recompute",
                "recompute_granularity",
                "sharding_stage",
            ]
            same = True
            for key in keys:
                if cfg[key] != cur_cfg[key]:
                    same = False
                    break
            if same:
                if "sharding_overlap" not in cfg:
                    raw_cfg = cfg
                elif not cfg["sharding_overlap"]:
                    if cfg["sharding_degree"] == cur_cfg["sharding_degree"]:
                        non_overlap_cfg = cfg
        assert non_overlap_cfg is not None
        assert raw_cfg is not None

        before_overlap_performance = non_overlap_cfg[
            tuner_cfg['metric_cfg']['name']
        ]
        overlap_performance = cur_cfg[tuner_cfg['metric_cfg']['name']]
        raw_performance = raw_cfg[tuner_cfg['metric_cfg']['name']]
        if (
            raw_performance
            and overlap_performance
            and before_overlap_performance
        ):
            ratio = (
                overlap_performance - before_overlap_performance
            ) / before_overlap_performance
            keys = copy.deepcopy(list(raw_cfg.keys()))
            for key in keys:
                if key.startswith("bw_") and raw_cfg[key]:
                    mew_key = "overlap_" + key
                    raw_cfg[mew_key] = round(raw_cfg[key] * (1 + ratio), 5)


def gen_sharding_overlap_args(res_args, cfg, tuner_cfg):
    """Generate args of sharding overlap."""
    if "sharding_overlap" not in tuner_cfg["search_algo"]:
        return
    cmd = copy.deepcopy(tuner_cfg["search_algo"]["sharding_overlap"])
    if cfg.get("sharding_overlap", False):
        valid_hybrid_strategy = ["sharding_mp", "sharding_pp", "sharding_mp_pp"]
        for key in cmd:
            if key not in valid_hybrid_strategy:
                raise ValueError(
                    f"Only support {valid_hybrid_strategy}, but got {key}."
                )
        sharding_degree = cfg["sharding_degree"]
        assert sharding_degree > 1
        mp_degree = cfg["mp_degree"]
        pp_degree = cfg["pp_degree"]
        arg = None
        if mp_degree > 1 and pp_degree == 1:
            arg = "sharding_mp"
        elif mp_degree == 1 and pp_degree > 1:
            arg = "sharding_pp"
        elif mp_degree > 1 and pp_degree > 1:
            arg = "sharding_mp_pp"
        else:
            return
        assert arg is not None
        if arg in cmd:
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))


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
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(cfg[arg]) if prefix else cfg[arg]
                    )
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))
        elif arg == "local_batch_size" and arg in cmd:
            global_batch_size = (
                cfg["global_batch_size"]
                if "global_batch_size"
                in tuner_cfg["model_cfg"]["global_batch_size"]
                else tuner_cfg["model_cfg"]["global_batch_size"]
            )
            local_batch_size = (
                global_batch_size // cfg["sharding_degree"] // cfg["dp_degree"]
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
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(local_batch_size)
                        if prefix
                        else local_batch_size
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(local_batch_size)
                        if prefix
                        else local_batch_size
                    )
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(local_batch_size)
                        if prefix
                        else local_batch_size
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(local_batch_size)
                        if prefix
                        else local_batch_size
                    )
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

        elif arg == "gradient_accumulation_steps" and arg in cmd:
            try:
                global_batch_size = (
                    cfg["global_batch_size"]
                    if "global_batch_size"
                    in tuner_cfg["model_cfg"]["global_batch_size"]
                    else tuner_cfg["model_cfg"]["global_batch_size"]
                )
                gradient_accumulation_steps = (
                    global_batch_size
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
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = json.load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(gradient_accumulation_steps)
                        if prefix
                        else gradient_accumulation_steps
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(gradient_accumulation_steps)
                        if prefix
                        else gradient_accumulation_steps
                    )
                json.dump(cmd_cfg, open(cmd[arg][0], "w"))
            elif ".yaml" in cmd[arg][0]:
                import yaml

                file_path = cmd[arg][0]
                prefix = ""
                if len(cmd[arg]) >= 3:
                    prefix = cmd[arg][2]
                try:
                    with open(file_path, "r") as f:
                        cmd_cfg = yaml.safe_load(f)
                except:
                    raise ValueError(
                        "Please check your auto tuner json whether valid."
                    )
                keys = cmd[arg][1].split(".")
                value = None
                for key in keys[: len(keys) - 1]:
                    if not value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = (
                        prefix + str(gradient_accumulation_steps)
                        if prefix
                        else gradient_accumulation_steps
                    )
                else:
                    cmd_cfg[keys[-1]] = (
                        prefix + str(gradient_accumulation_steps)
                        if prefix
                        else gradient_accumulation_steps
                    )
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = cmd_cfg[key]
                    else:
                        value = value[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
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
                value = None
                for key in keys[: len(keys) - 1]:
                    if value:
                        value = value[key]
                    else:
                        value = cmd_cfg[key]
                if value:
                    value[keys[-1]] = cmd[arg][2]
                else:
                    cmd_cfg[keys[-1]] = cmd[arg][2]
                yaml.dump(cmd_cfg, open(cmd[arg][0], "w"))

    # sharding overlap args
    gen_sharding_overlap_args(res_args, cfg, tuner_cfg)

    return res_args


def gen_new_ctx(ctx, cur_cfg, tuner_cfg):
    """Generate new running context."""
    new_ctx = copy.deepcopy(ctx)
    if (
        "search_algo" in tuner_cfg
        and "estimated_num_gpus" in tuner_cfg["search_algo"]
    ):
        assert cur_cfg["dp_degree"] == 1
        assert cur_cfg["sharding_stage"] == 1
        actual_cards = (
            cur_cfg["mp_degree"]
            * cur_cfg["pp_degree"]
            * cur_cfg["sharding_degree"]
        )
        if actual_cards <= tuner_cfg["gpus_per_node"]:
            new_ctx.args.devices = ",".join(
                [str(i) for i in range(actual_cards)]
            )
            if new_ctx.args.master:
                new_ctx.args.nnodes = "1:1"
        else:
            new_ctx.args.devices = ",".join(
                [str(i) for i in range(tuner_cfg["gpus_per_node"])]
            )
            nnodes = actual_cards // tuner_cfg["gpus_per_node"]
            new_ctx.args.nnodes = f"{nnodes}:{nnodes}"
    return new_ctx


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


def read_step_time_log(
    path, file="workerlog.0", target_metric='interval_runtime'
) -> Tuple[float, int]:
    target_file = path + "/" + file
    if not os.path.exists(target_file):
        return None
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
                value = None
                for item in metric[0]:
                    try:
                        value = float(item)
                        metric_list.append(value)
                        break
                    except:
                        continue
                assert value is not None
        if not metric_list:
            metric_ave = None
            return None
        elif len(metric_list) < 10:
            metric_ave = metric_list[-1]
        elif len(metric_list) < 20:
            metric_ave = sum(metric_list[9:]) / (len(metric_list[9:]))
        else:
            metric_ave = sum(metric_list[-10:]) / 10
        # round to 5 decimal places
        metric_ave = round(metric_ave, 5)
    res = metric_ave
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


def load_configs_from_csv(configs_csv):
    """Load the configs from csv file."""
    all_configs = []
    extract_keys_integer = [
        "dp_degree",
        "mp_degree",
        "pp_degree",
        "vpp_degree",
        "micro_batch_size",
        "sharding_degree",
        "sharding_stage",
    ]
    extract_keys_string = ["use_recompute", "recompute_granularity"]
    with open(configs_csv, "r") as f:
        reader = csv.DictReader(f)
        raw_configs = list(reader)
    for raw_config in raw_configs:
        config = {}
        for extract_key in extract_keys_integer:
            val = raw_config.get(extract_key, "")
            try:
                config[extract_key] = int(val)
            except ValueError:
                raise ValueError(
                    f"{extract_key} must be integer, but got {val}"
                )

        use_recompute = raw_config.get("use_recompute", "")
        assert use_recompute.lower() in [
            "true",
            "false",
        ], f"{use_recompute} must be true or false, but got {use_recompute}"
        config["use_recompute"] = use_recompute.lower() == "true"

        recompute_granularity = raw_config.get("recompute_granularity", "")
        assert (
            recompute_granularity == ""
            or recompute_granularity.lower()
            in __SUPPORTED_RECOMPUTE_GRANULARITY__
        ), f"{recompute_granularity} must be one of {__SUPPORTED_RECOMPUTE_GRANULARITY__}, but got {recompute_granularity}."
        config["recompute_granularity"] = (
            recompute_granularity if recompute_granularity != "" else None
        )

        all_configs.append(config)

    return all_configs
