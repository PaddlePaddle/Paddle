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

_PRUNE_FUNC = []


def same_cfgs_beside(attr, cur_cfg, history_cfgs):
    """
    Compare the current configuration with the history configuration,
    and obtain the same configurations as the current configuration except for the given attr.
    """
    results = []
    same = True
    for cfg in history_cfgs:
        for key in cur_cfg:
            if key == attr:
                continue
            if key not in cfg or cfg[key] != cur_cfg[key]:
                same = False
                break
        if same:
            results.append(cfg)
        else:
            same = True
    return results


def register_prune(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    _PRUNE_FUNC.append(wrapper)
    return wrapper


@register_prune
def prune_by_mp(tuner_cfg, cur_cfg, history_cfgs=None):
    """
    Prune by mp, the rules are:
    1. MP degree should be evenly divided by hidden size and vocab size
    2. MP degree should be in the candidates of user defined.
    3. MP degree should be less than 8 if no candidates.
    """
    mp_degree = cur_cfg.get("mp_degree", None)
    hidden_size = tuner_cfg["model_cfg"].get("hidden_size", None)
    vocab_size = tuner_cfg["model_cfg"].get("vocab_size", None)

    if mp_degree is None:
        return False

    if hidden_size and hidden_size % mp_degree != 0:
        return True

    if vocab_size and vocab_size % mp_degree != 0:
        return True

    mp_degree_candidates = tuner_cfg.get("mp_degree", None)

    if mp_degree_candidates == "auto":
        mp_degree_candidates = tuner_cfg["candidates"]["mp_degree"]

    if mp_degree_candidates:
        if mp_degree not in mp_degree_candidates:
            return True

    # prune default candidates
    if mp_degree > 8:
        return True

    return False


@register_prune
def prune_by_pp(tuner_cfg, cur_cfg, history_cfgs=None):
    """
    Prune by pp (pipeline-parallelism), the rules are:
    1. PP degree should be evenly divided by number of layers.
    2. PP degree should be in the candidates of user defined.
    3. If no candidates, PP degree should be less than or equal to the number of nodes.
    """
    pp_degree = cur_cfg.get("pp_degree", None)
    num_layers = tuner_cfg["model_cfg"].get("num_layers", None)
    num_nodes = tuner_cfg.get("nodes", 1)

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


@register_prune
def prune_by_mbs(tuner_cfg, cur_cfg, history_cfgs=None):
    """
    Prune by mbs (micro batch size), the rules are:
    1. Micro batch size should be evenly divided by the local batch size.
    2. Micro batch size should be in the candidates of user defined.
    3. Prune if a similar configuration with a larger micro batch size resulted in a valid run.
    """
    micro_batch_size = cur_cfg.get("micro_batch_size", None)
    global_batch_size = tuner_cfg["model_cfg"].get("global_batch_size", None)
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

    if mbs_candidates:
        if micro_batch_size not in mbs_candidates:
            return True

    cfgs = same_cfgs_beside("micro_batch_size", cur_cfg, history_cfgs)
    if cfgs:
        for cfg in cfgs:
            if (
                cfg["micro_batch_size"] > micro_batch_size
                and cfg.get("time", -1) > 0
            ):
                return True

    return False


@register_prune
def prune_by_sharding(tuner_cfg, cur_cfg, history_cfgs):
    """
    Prune by sharding parameters, the rules are:
    1. Sharding stage and sharding degree should be specified.
    2. Sharding stage and degree should be in the candidates of user defined.
    3. If PP (pipeline-parallelism) degree is not 1, sharding stage must be 1.
    4. Prune if a similar configuration with a lower sharding stage resulted in a valid run.
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

    if pp_degree and pp_degree != 1 and sharding_stage != 1:
        return True

    cfgs = same_cfgs_beside("sharding_stage", cur_cfg, history_cfgs)
    if cfgs:
        for cfg in cfgs:
            if (
                cfg["sharding_stage"] < sharding_stage
                and cfg.get("time", -1) > 0
            ):
                return True

    if sharding_degree == 1:
        cfgs = same_cfgs_beside("sharding_stage", cur_cfg, history_cfgs)
        if cfgs:
            return True

    return False


@register_prune
def prune_by_recompute(tuner_cfg, cur_cfg, history_cfgs):
    """
    Prune by recompute parameters, the rules are:
    1. If recompute is not used, return False directly.
    2. Usage of recompute and recompute granularity should be in the candidates of user defined.
    3. If recompute is not used, but recompute granularity is set, return True for pruning.
    4. Prune if a similar configuration without using recompute resulted in a valid run.
    """
    recompute_granularity = cur_cfg.get("recompute_granularity", None)
    use_recompute = cur_cfg.get("use_recompute", None)
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

    if not use_recompute and recompute_granularity:
        return True

    cfgs = same_cfgs_beside("use_recompute", cur_cfg, history_cfgs)
    if cfgs:
        for cfg in cfgs:
            if (
                not cfg["use_recompute"]
                and use_recompute
                and cfg.get("time", -1) > 0
            ):
                return True

    if not use_recompute:
        cfgs = same_cfgs_beside("recompute_granularity", cur_cfg, history_cfgs)
        if cfgs:
            return True

    return False


@register_prune
def prune_by_num_gpus(tuner_cfg, cur_cfg, history_cfgs):
    num_gpus = tuner_cfg.get("num_gpus")
    dp_degree = cur_cfg.get("dp_degree", 1)
    mp_degree = cur_cfg.get("mp_degree", 1)
    pp_degree = cur_cfg.get("pp_degree", 1)
    sharding_degree = cur_cfg.get("sharding_degree", 1)
    if dp_degree * mp_degree * pp_degree * sharding_degree != num_gpus:
        return True

    return False
