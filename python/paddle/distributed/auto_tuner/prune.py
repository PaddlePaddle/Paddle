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
import logging
import os
import subprocess

logger = logging.getLogger('auto_tuner')
_PRUNE_FUNC = []
_PRUNE_HISTORY_FUNC = []


def log_pruned_info(cur_cfg, pruned_reason, tuner_cfg):
    pruned_strategy = "DP{}_MP{}_PP{}_VPP{}_Sharding{}_Stage{}_MBS{}_Recompute_{}_Granularity_{}".format(
        cur_cfg["dp_degree"],
        cur_cfg["mp_degree"],
        cur_cfg["pp_degree"],
        cur_cfg["vpp_degree"],
        cur_cfg["sharding_degree"],
        cur_cfg["sharding_stage"],
        cur_cfg["micro_batch_size"],
        cur_cfg["use_recompute"],
        cur_cfg["recompute_granularity"],
    )
    if "refined_recompute" in tuner_cfg:
        for key in tuner_cfg["refined_recompute"]:
            strategy = "".join(i.capitalize() for i in key.split("_"))
            strategy += str(cur_cfg[key])
            pruned_strategy = pruned_strategy + "_" + strategy

    if "custom_search_dim" in tuner_cfg:
        for key in tuner_cfg["custom_search_dim"]:
            strategy = "".join(i.capitalize() for i in key.split("_"))
            strategy += str(cur_cfg[key])
            pruned_strategy = pruned_strategy + "_" + strategy

    try:
        from paddle.distributed.launch.main import ctx

        ctx.logger.info(
            f"Strategy {pruned_strategy} has been pruned that {pruned_reason}"
        )
    except:
        pass
    logger.info(
        f"Strategy {pruned_strategy} has been pruned that {pruned_reason}"
    )


def same_cfgs_beside(attrs, cur_cfg, history_cfgs=[]):
    """
    Compare the current configuration with the history configuration,
    and obtain the same configurations as the current configuration except for the given attr.
    """
    results = []
    same = True

    for cfg in history_cfgs:
        for key in cur_cfg:
            if key in attrs:
                continue
            if key not in cfg or (
                cfg[key] != cur_cfg[key]
                and key not in ["estimated_memory_usage"]
            ):
                same = False
                break
        if same:
            results.append(cfg)
        else:
            same = True

    return results


def same_cfgs_beside_sharding_overlap(tuner_cfg, cur_cfg, history_cfgs=[]):
    result = None
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
            result = cfg
            break
    return result


def register_prune(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _PRUNE_FUNC.append(wrapper)
    return wrapper


def register_prune_history(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _PRUNE_HISTORY_FUNC.append(wrapper)
    return wrapper


@register_prune
def prune_by_mp(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by mp, the rules are:
    1. MP degree should be evenly divided by hidden size and vocab size
    2. MP degree should be in the candidates of user defined.
    3. MP degree should be less than 8 if no candidates.
    """
    mp_degree = cur_cfg.get("mp_degree", None)
    hidden_size = tuner_cfg["model_cfg"].get("hidden_size", None)
    vocab_size = tuner_cfg["model_cfg"].get("vocab_size", None)
    num_attention_heads = tuner_cfg["model_cfg"].get(
        "num_attention_heads", None
    )
    seq_length = tuner_cfg["model_cfg"].get("seq_length", None)
    use_sequence_parallel = tuner_cfg.get("use_sequence_parallel", False)

    if mp_degree is None:
        return False

    if hidden_size and hidden_size % mp_degree != 0:
        return True

    if vocab_size and vocab_size % mp_degree != 0:
        return True

    if num_attention_heads and num_attention_heads % mp_degree != 0:
        return True

    if seq_length and seq_length % mp_degree != 0 and use_sequence_parallel:
        return True

    mp_degree_candidates = tuner_cfg.get("mp_degree", None)

    if mp_degree_candidates == "auto":
        mp_degree_candidates = tuner_cfg["candidates"]["mp_degree"]

    if mp_degree_candidates:
        if mp_degree not in mp_degree_candidates:
            return True

    return False


@register_prune
def prune_by_pp(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by pp (pipeline-parallelism), the rules are:
    1. PP degree should be evenly divided by number of layers.
    2. PP degree should be in the candidates of user defined.
    3. If no candidates, PP degree should be less than or equal to the number of nodes.
    """
    pp_degree = cur_cfg.get("pp_degree", None)
    num_layers = tuner_cfg["model_cfg"].get("num_layers", None)
    num_nodes = (
        cur_cfg["nodes"] if "nodes" in cur_cfg else tuner_cfg.get("nodes", 1)
    )

    if pp_degree is None:
        return False

    if num_layers:
        if num_layers % pp_degree != 0:
            return True

    pp_degree_candidates = tuner_cfg.get("pp_degree", None)
    if pp_degree_candidates == "auto":
        pp_degree_candidates = tuner_cfg["candidates"]["pp_degree"]
    if pp_degree_candidates:
        if pp_degree not in pp_degree_candidates:
            return True
    else:
        if num_nodes != 1 and pp_degree > num_nodes:
            return True
    return False


@register_prune_history
def prune_by_mp_pp_history(tuner_cfg, cur_cfg, history_cfgs, pruned_cfgs):
    mp_degree = cur_cfg.get("mp_degree", None)
    pp_degree = cur_cfg.get("pp_degree", None)
    use_recompute = cur_cfg.get("recompute", None)

    if mp_degree is None or pp_degree is None or use_recompute is None:
        return False
    history_cfgs = copy.deepcopy(history_cfgs)
    history_cfgs.extend(pruned_cfgs)
    cfgs = same_cfgs_beside(["mp_degree", "pp_degree"], cur_cfg, history_cfgs)

    if cfgs:
        for cfg in cfgs:
            if (
                not use_recompute
                and cfg["mp_degree"] * cfg["pp_degree"] == mp_degree * pp_degree
                and cfg["mp_degree"] > mp_degree
                and cfg.get("max_mem_usage") == "OOM"
            ):
                pruned_reason = f"mp_degree {mp_degree}, pp_degree {pp_degree} may cause oom because {cfg['mp_degree']}, {cfg['pp_degree']} already oom."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["max_mem_usage"] = "OOM"
                return True

    return False


@register_prune
def prune_by_vpp(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by vpp (virtual pipeline parallelism), the rules are:
    1. VPP degree should be evenly divided by number of layers.
    2. VPP degree should be in the candidates of user defined.
    """
    pp_degree = cur_cfg.get("pp_degree", None)
    vpp_degree = cur_cfg.get("vpp_degree", None)
    num_layers = tuner_cfg["model_cfg"].get("num_layers", None)

    if pp_degree is None:
        return False

    if vpp_degree is None:
        return False

    if num_layers:
        global_batch_size = (
            cur_cfg["global_batch_size"]
            if "global_batch_size" in cur_cfg
            else tuner_cfg["model_cfg"].get("global_batch_size", None)
        )
        acc_steps = (
            global_batch_size
            // cur_cfg["dp_degree"]
            // cur_cfg["sharding_degree"]
            // cur_cfg["micro_batch_size"]
        )
        if vpp_degree > 1 and acc_steps % pp_degree != 0:
            return True
        if num_layers % (pp_degree * vpp_degree) != 0:
            return True
        if pp_degree == 1 and vpp_degree != 1:
            return True
        if pp_degree <= 2 and vpp_degree != 1:
            return True

    vpp_degree_candidates = tuner_cfg.get("vpp_degree", None)
    if vpp_degree_candidates == "auto":
        vpp_degree_candidates = tuner_cfg["candidates"]["vpp_degree"]
    if vpp_degree_candidates:
        if vpp_degree not in vpp_degree_candidates:
            return True

    return False


@register_prune_history
def prune_by_vpp_history(tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]):
    vpp_degree = cur_cfg.get("vpp_degree", None)
    if vpp_degree is None:
        return False
    history_cfgs = copy.deepcopy(history_cfgs)
    history_cfgs.extend(pruned_cfgs)

    cfgs = same_cfgs_beside("vpp_degree", cur_cfg, history_cfgs)

    if cfgs:
        for cfg in cfgs:
            # memory prune
            if (
                cfg["vpp_degree"] > vpp_degree
                and cfg.get("max_mem_usage") == "OOM"
            ):
                pruned_reason = f"vpp_degree {vpp_degree} may cause oom because { cfg['vpp_degree']} already oom."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["max_mem_usage"] = "OOM"
                return True

    return False


@register_prune
def prune_by_mbs(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by mbs (micro batch size), the rules are:
    1. Micro batch size should be evenly divided by the local batch size.
    2. Micro batch size should be in the candidates of user defined.
    3. Prune if a similar configuration with a larger micro batch size resulted in a valid run.
    """
    micro_batch_size = cur_cfg.get("micro_batch_size", None)
    global_batch_size = (
        cur_cfg["global_batch_size"]
        if "global_batch_size" in cur_cfg
        else tuner_cfg["model_cfg"].get("global_batch_size", None)
    )
    if global_batch_size == "auto":
        global_batch_size = cur_cfg["global_batch_size"]
    if global_batch_size:
        local_batch_size = (
            global_batch_size
            // cur_cfg["dp_degree"]
            // cur_cfg["sharding_degree"]
        )
        if local_batch_size == 0:
            return True

    mbs_candidates = tuner_cfg.get("micro_batch_size", None)

    if mbs_candidates == "auto":
        mbs_candidates = tuner_cfg["candidates"]["micro_batch_size"]

    if micro_batch_size is None:
        return False

    if local_batch_size:
        if local_batch_size % micro_batch_size != 0:
            return True
        acc_steps = local_batch_size // micro_batch_size
        pp_degree = cur_cfg.get("pp_degree", None)
        if pp_degree is not None:
            if acc_steps < pp_degree:
                return True
        vpp_degree = cur_cfg.get("vpp_degree", None)
        if vpp_degree is not None and vpp_degree > 1:
            if pp_degree is not None:
                if acc_steps % pp_degree != 0:
                    return True

    if mbs_candidates:
        if micro_batch_size not in mbs_candidates:
            return True

    return False


@register_prune_history
def prune_by_mbs_history(tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]):
    micro_batch_size = cur_cfg.get("micro_batch_size", None)
    if micro_batch_size is None:
        return False
    history_cfgs = copy.deepcopy(history_cfgs)
    history_cfgs.extend(pruned_cfgs)

    cfgs = same_cfgs_beside(
        ["micro_batch_size", "acc_steps"], cur_cfg, history_cfgs
    )

    if cfgs:
        for cfg in cfgs:
            if (
                cfg["micro_batch_size"] > micro_batch_size
                and cfg.get("time", -1) > 0
            ):
                pruned_reason = f"micro_batch_size {micro_batch_size} may be slower because {cfg['micro_batch_size']} has been already runnable."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["time"] = cfg["time"]
                return True
            # memory prune
            if (
                cfg["micro_batch_size"] < micro_batch_size
                and cfg.get("max_mem_usage") == "OOM"
            ):
                pruned_reason = f"micro_batch_size {micro_batch_size} may cause oom because {cfg['micro_batch_size']} already oom."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["max_mem_usage"] = "OOM"
                return True
    return False


@register_prune
def prune_by_sharding(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by sharding parameters, the rules are:
    1. Sharding stage and sharding degree should be specified.
    2. Sharding stage and degree should be in the candidates of user defined.
    3. If PP (pipeline-parallelism) degree is not 1, sharding stage must be 1.
    4. Prune if a similar configuration with a lower sharding stage resulted in a valid run.
    5. If sharding degree is 1, sharding stage is invalid.
    """
    sharding_stage = cur_cfg.get("sharding_stage", None)
    sharding_degree = cur_cfg.get("sharding_degree", None)
    pp_degree = cur_cfg.get("pp_degree", None)

    if not sharding_stage:
        return False

    if not sharding_degree:
        return False

    sharding_stage_candidates = tuner_cfg.get("sharding_stage", None)
    if sharding_stage_candidates == "auto":
        sharding_stage_candidates = tuner_cfg["candidates"]["sharding_stage"]

    sharding_degree_candidates = tuner_cfg.get("sharding_degree", None)
    if sharding_degree_candidates == "auto":
        sharding_degree_candidates = tuner_cfg["candidates"]["sharding_degree"]

    if sharding_stage_candidates:
        if sharding_stage not in sharding_stage_candidates:
            return True

    if sharding_degree_candidates:
        if sharding_degree not in sharding_degree_candidates:
            return True

    if (
        pp_degree
        and pp_degree != 1
        and sharding_stage != 1
        and sharding_degree != 1
    ):
        return True

    if sharding_degree == 1:
        cfgs = same_cfgs_beside("sharding_stage", cur_cfg, history_cfgs)
        if cfgs:
            return True

    return False


@register_prune_history
def prune_by_sharding_history(
    tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]
):
    sharding_degree = cur_cfg.get("sharding_degree", None)
    if sharding_degree is None:
        return False

    sharding_stage = cur_cfg.get("sharding_stage", None)
    if sharding_stage is None:
        return False
    history_cfgs = copy.deepcopy(history_cfgs)
    history_cfgs.extend(pruned_cfgs)

    cfgs = same_cfgs_beside("sharding_stage", cur_cfg, history_cfgs)
    if cfgs:
        for cfg in cfgs:
            if (
                cfg["sharding_stage"] < sharding_stage
                and cfg.get("time", -1) > 0
            ):
                pruned_reason = f"sharding_stage {sharding_stage} may be slower because {cfg['sharding_stage'] } has been already runnable."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["time"] = cfg["time"]
                return True

            # memory prune
            if (
                cfg["sharding_stage"] > sharding_stage
                and cfg.get("max_mem_usage") == "OOM"
            ):
                pruned_reason = f"sharding_stage {sharding_stage} may cause oom because {cfg['sharding_stage']} already oom."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["max_mem_usage"] = "OOM"
                return True

    return False


@register_prune
def prune_by_recompute(tuner_cfg, cur_cfg, history_cfgs=[]):
    """
    Prune by recompute parameters, the rules are:
    1. If recompute is not used, return False directly.
    2. Usage of recompute and recompute granularity should be in the candidates of user defined.
    3. If recompute is not used, but recompute granularity is set, return True for pruning.
    4. Prune if a similar configuration without using recompute resulted in a valid run.
    5. If recompute is false, prune redundant recompute granularity
    """
    recompute_granularity = cur_cfg.get("recompute_granularity", None)
    use_recompute = cur_cfg.get("use_recompute", None)
    recompute_level = get_config_recompute_level(cur_cfg)

    if use_recompute is None:
        return False

    recompute_granularity_candidates = tuner_cfg["candidates"].get(
        "recompute_granularity", None
    )
    use_recompute_candidates = tuner_cfg["candidates"].get(
        "use_recompute", None
    )

    if use_recompute_candidates:
        if use_recompute not in use_recompute_candidates:
            return True

    if recompute_granularity_candidates and recompute_granularity:
        if recompute_granularity not in recompute_granularity_candidates:
            return True

    if not use_recompute:
        if recompute_granularity != "full":
            return True

        cfgs = same_cfgs_beside(
            ["use_recompute", "recompute_granularity"], cur_cfg, history_cfgs
        )
        if cfgs:
            for cfg in cfgs:
                if recompute_level == get_config_recompute_level(cfg):
                    return True

    return False


def get_config_recompute_level(cfg):
    recompute_granularity_level = {"full": 3, "full_attn": 2, "core_attn": 1}
    use_recompute = cfg.get("use_recompute", None)
    recompute_granularity = cfg.get("recompute_granularity", None)

    if use_recompute is None:
        return None

    if not use_recompute:
        return 0
    else:
        return recompute_granularity_level[recompute_granularity]


@register_prune_history
def prune_by_recompute_history(
    tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]
):
    recompute_level = get_config_recompute_level(cur_cfg)

    if recompute_level is None:
        return False

    history_cfgs = copy.deepcopy(history_cfgs)
    history_cfgs.extend(pruned_cfgs)

    cfgs = same_cfgs_beside(
        ["use_recompute", "recompute_granularity"], cur_cfg, history_cfgs
    )

    if cfgs:
        for cfg in cfgs:
            cfg["recompute_level"] = get_config_recompute_level(cfg)

            if (
                cfg["recompute_level"] < recompute_level
                and cfg.get("time", -1) > 0
            ):
                pruned_reason = f"use_recompute may be slower because {cfg['use_recompute']} has been already runnable."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["time"] = cfg["time"]
                return True

            if (
                cfg["recompute_level"] > recompute_level
                and cfg.get("max_mem_usage") == "OOM"
            ):
                pruned_reason = f"use_recompute may cause oom because {cfg['use_recompute']} already oom."
                log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                cur_cfg["max_mem_usage"] = "OOM"
                return True

    return False


@register_prune
def prune_by_num_gpus(tuner_cfg, cur_cfg, history_cfgs=[]):
    num_gpus = (
        cur_cfg["num_gpus"]
        if "num_gpus" in cur_cfg
        else tuner_cfg.get("num_gpus")
    )
    dp_degree = cur_cfg.get("dp_degree", 1)
    mp_degree = cur_cfg.get("mp_degree", 1)
    pp_degree = cur_cfg.get("pp_degree", 1)
    sharding_degree = cur_cfg.get("sharding_degree", 1)
    if dp_degree * mp_degree * pp_degree * sharding_degree != num_gpus:
        return True

    return False


@register_prune
def prune_by_memory_estimation(tuner_cfg, cur_cfg, history_cfgs=[]):
    memory_estimation_tool = tuner_cfg.get("memory_estimation_tool", None)
    # TODO(@gexiao): get from system api
    max_memory_usage = tuner_cfg.get("max_mem_usage", None)
    model_cfg = tuner_cfg["model_cfg"]

    if memory_estimation_tool is None:
        return False

    if not os.path.exists(memory_estimation_tool):
        raise ValueError(
            f"memory_estimation_tool should be a valid path, but got {memory_estimation_tool}"
        )

    if max_memory_usage is None:
        raise ValueError(
            "max_mem_usage should be set when using memory estimation tool"
        )

    # get distributed strategy
    dp_degree = cur_cfg['dp_degree']
    mp_degree = cur_cfg['mp_degree']
    pp_degree = cur_cfg['pp_degree']
    vpp_degree = cur_cfg['vpp_degree']
    sharding_degree = cur_cfg['sharding_degree']
    sharding_stage = cur_cfg['sharding_stage']
    use_recompute = cur_cfg['use_recompute']
    micro_batch_size = cur_cfg['micro_batch_size']
    recompute_granularity = cur_cfg['recompute_granularity']

    memory_estimation_cmd = [
        "python",
        memory_estimation_tool,
        "--dp_degree",
        str(dp_degree),
        "--mp_degree",
        str(mp_degree),
        "--pp_degree",
        str(pp_degree),
        "--vpp_degree",
        str(vpp_degree),
        "--sharding_degree",
        str(sharding_degree),
        "--sharding_stage",
        str(sharding_stage),
        "--use_recompute",
        str(use_recompute),
        "--micro_batch_size",
        str(micro_batch_size),
        "--recompute_granularity",
        str(recompute_granularity),
    ]

    # get model config
    hidden_size = model_cfg.get('hidden_size', None)
    if hidden_size is not None:
        memory_estimation_cmd.extend(["--hidden_size", str(hidden_size)])

    num_attention_heads = model_cfg.get('num_attention_heads', None)
    if num_attention_heads is not None:
        memory_estimation_cmd.extend(
            ["--num_attention_heads", str(num_attention_heads)]
        )

    num_layers = model_cfg.get('num_layers', None)
    if num_layers is not None:
        memory_estimation_cmd.extend(["--num_layers", str(num_layers)])

    max_sequence_length = model_cfg.get('max_sequence_length', None)
    if max_sequence_length is not None:
        memory_estimation_cmd.extend(
            ["--max_sequence_length", str(max_sequence_length)]
        )

    vocab_size = model_cfg.get('vocab_size', None)
    if vocab_size is not None:
        memory_estimation_cmd.extend(["--vocab_size", str(vocab_size)])

    intermediate_size = model_cfg.get('intermediate_size', None)
    if intermediate_size is not None:
        memory_estimation_cmd.extend(
            ["--intermediate_size", str(intermediate_size)]
        )

    result = subprocess.run(
        memory_estimation_cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        cur_memory_usage = int(round(float(result.stdout), 2))
        cur_cfg["estimated_memory_usage"] = cur_memory_usage
        msg = f"Estimated {cur_cfg} memory usage: {cur_memory_usage} MB"
        memory_exceeded = cur_memory_usage > (max_memory_usage * 1024)
        if memory_exceeded:
            msg += ", it will be pruned!"
        logger.info(msg)
        return memory_exceeded
    else:
        raise ValueError(
            f"memory_estimation_tool failed with error: {result.stderr}"
        )


@register_prune_history
def prune_by_sharding_overlap(
    tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]
):
    """Prune by sharding overlap for single dp estimation"""
    if "sharding_overlap" in cur_cfg:
        result = same_cfgs_beside_sharding_overlap(
            tuner_cfg, cur_cfg, history_cfgs
        )
        if not result:
            return True
        if not result[tuner_cfg['metric_cfg']['name']]:
            return True
    return False


def is_invalid(cur_cfg, invalid_strategy):
    mapping = {
        "dp_degree": "dp",
        "mp_degree": "mp",
        "pp_degree": "pp",
        "vpp_degree": "vpp",
        "micro_batch_size": "mbs",
        "sharding_degree": "sharding",
        "sharding_stage": "stage",
        "use_recompute": "recompute",
        "recompute_granularity": "granularity",
    }
    granularity_mapping = {0: "full", 1: "full_attn", 2: "core_attn"}
    reversed_mapping = {}
    for key in mapping:
        reversed_mapping[mapping[key]] = key

    for strategy in invalid_strategy:
        assert isinstance(strategy, str)
        dims = strategy.split("_")
        has_matched = 0
        for dim in dims:
            matched = None
            for key in reversed_mapping:
                if dim.startswith(key):
                    matched = key
                    break
            if matched:
                value = dim[len(matched)]
                # * means this strategy turned on
                if matched in ["dp", "mp", "pp", "vpp", "sharding"]:
                    if value == "*":
                        if cur_cfg[reversed_mapping[matched]] != 1:
                            has_matched += 1
                            continue
                    else:
                        value = int(value)
                        if cur_cfg[reversed_mapping[matched]] == value:
                            has_matched += 1
                            continue
                elif matched == "recompute":
                    if value == "*":
                        if cur_cfg[reversed_mapping[matched]]:
                            has_matched += 1
                            continue
                    else:
                        value = bool(int(value))
                        if cur_cfg[reversed_mapping[matched]] == value:
                            has_matched += 1
                            continue
                elif matched == "stage":
                    if value == "*":
                        if cur_cfg[reversed_mapping["sharding"]] != 1:
                            has_matched += 1
                            continue
                    else:
                        value = int(value)
                        if cur_cfg[reversed_mapping[matched]] == value:
                            has_matched += 1
                            continue
                elif matched == "mbs":
                    if value == "*":
                        has_matched += 1
                        continue
                    else:
                        value = int(value)
                        if cur_cfg[reversed_mapping[matched]] == value:
                            has_matched += 1
                            continue
                elif matched == "granularity":
                    if value == "*":
                        if cur_cfg[reversed_mapping["use_recompute"]]:
                            has_matched += 1
                            continue
                    else:
                        value = int(value)
                        granularity = granularity_mapping[value]
                        if cur_cfg[reversed_mapping[matched]] == granularity:
                            has_matched += 1
                            continue
    if has_matched == len(dims):
        return True
    return False


@register_prune
def prune_by_invalid_strategy(tuner_cfg, cur_cfg, history_cfgs=[]):
    if tuner_cfg.get("invalid_strategy", None):
        invalid_strategy = tuner_cfg["invalid_strategy"]
        assert isinstance(invalid_strategy, list)
        if is_invalid(cur_cfg, invalid_strategy):
            return True

    return False


@register_prune
def prune_by_refined_recompute(tuner_cfg, cur_cfg, history_cfgs=[]):
    if tuner_cfg.get("refined_recompute", None):
        rr = tuner_cfg.get("refined_recompute")
        pp_degree = cur_cfg["pp_degree"]
        recompute = cur_cfg["use_recompute"]
        recompute_granularity = cur_cfg["recompute_granularity"]
        compare = [cur_cfg[item] for item in rr]
        if recompute:
            if recompute_granularity and recompute_granularity != "full":
                if compare.count(0) != len(compare):
                    return True
        if pp_degree == 1 and compare.count(0) != len(compare):
            return True
        if tuner_cfg["model_cfg"]["num_layers"] % pp_degree != 0:
            return True
        max_value = tuner_cfg["model_cfg"]["num_layers"] / pp_degree
        if cur_cfg[rr[0]] > max_value:
            return True
        i = 1
        while i < len(rr):
            if cur_cfg[rr[i]] > max_value or (
                cur_cfg[rr[i - 1]] != max_value and cur_cfg[rr[i]] != 0
            ):
                return True
            i += 1

    return False


@register_prune_history
def prune_by_refined_recompute_history(
    tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]
):
    if tuner_cfg.get("refined_recompute", None):
        history_cfgs = copy.deepcopy(history_cfgs)
        history_cfgs.extend(pruned_cfgs)
        rr = tuner_cfg.get("refined_recompute")
        compare = copy.deepcopy(rr)
        compare.append("use_recompute")
        cfgs = same_cfgs_beside(compare, cur_cfg, history_cfgs)
        for item in rr:
            if cfgs:
                for cfg in cfgs:
                    if not cfg["use_recompute"] and cfg.get("time", -1) > 0:
                        pruned_reason = f"{item} {cur_cfg[item]} may be slower because not recompute has been already runnable."
                        log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                        cur_cfg["time"] = cfg["time"]
                        return True
                    if (
                        cfg[item] > cur_cfg[item]
                        and cfg.get("time", -1) > 0
                        and cfg["use_recompute"]
                        and cur_cfg["use_recompute"]
                    ):
                        pruned_reason = f"{item} {cur_cfg[item]} may be slower because {cfg[item]} has been already runnable."
                        log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                        cur_cfg["time"] = cfg["time"]
                        return True
                    # memory prune
                    if (
                        cfg[item] < cur_cfg[item]
                        and cfg.get("max_mem_usage") == "OOM"
                        and cfg["use_recompute"]
                        and cur_cfg["use_recompute"]
                    ):
                        pruned_reason = f"{item} {cur_cfg[item]} may cause oom because {cfg[item]} already oom."
                        log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                        cur_cfg["max_mem_usage"] = "OOM"
                        return True

    return False


@register_prune_history
def prune_by_custom_search_dim_history(
    tuner_cfg, cur_cfg, history_cfgs=[], pruned_cfgs=[]
):
    history_cfgs = copy.deepcopy(history_cfgs)
    custom_search_dim = tuner_cfg.get("custom_search_dim", None)
    prune_custom_search_dim = []
    custom_dim_level = {}
    if custom_search_dim is not None:
        for key, value in custom_search_dim.items():
            if value["prune"]:
                prune_custom_search_dim.append(key)
                # In the custom_search_dim, the values are ordered according to the sequence specified in its custom configuration.
                custom_dim_level[key] = {
                    key: value for value, key in enumerate(value["value"])
                }

    for key in prune_custom_search_dim:
        history_cfgs.extend(pruned_cfgs)
        cfgs = same_cfgs_beside(key, cur_cfg, history_cfgs)
        cur_value = cur_cfg.get(key, None)
        if cur_value is None:
            return False

        # In the custom_search_dim, based on the order of values provided in its custom configuration, if a configuration is found to be executable, the subsequent configurations will be pruned.
        if cfgs:
            for cfg in cfgs:
                cfg_value = cfg[key]
                if (
                    custom_dim_level[key][cfg_value]
                    < custom_dim_level[key][cur_value]
                    and cfg.get("time", -1) > 0
                ):
                    pruned_reason = f"{key}{cfg_value} may be slower because {key}{cur_value} has been already runnable."
                    log_pruned_info(cur_cfg, pruned_reason, tuner_cfg)
                    cur_cfg["time"] = cfg["time"]
                    return True

    return False
