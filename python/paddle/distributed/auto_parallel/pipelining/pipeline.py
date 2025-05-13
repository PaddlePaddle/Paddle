import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.utils import flatten
from schedules import PipelineScheduleSingle
import functools
from paddle.distributed import Replicate
from stage import PipelineStage


class _Pipeline_model_chunk(nn.Layer):
    def __init__(self, layers):
        if not isinstance(layers, (list, tuple)):
            raise TypeError(
                f"Expected type of `layers` to be a list|tuple but got {type(layers)}."
            )
        self.layers = layers
        super(_Pipeline_model_chunk, self).__init__()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _get_style(schedule_mode):
        # TODO 
        return None


def _flatten_layers(layers):
    tmp_layers = []
    for layer in layers:
        if isinstance(layer, nn.LayerList):
            for tmp in layer:
                tmp_layers.append(tmp)
        else:
            tmp_layers.append(layer)
    return tmp_layers


def get_pp_mesh(pp_idx=None):
    """
    获得pp_idx的mesh
    """
    mesh = dist.fleet.auto.get_mesh()
    if pp_idx is not None and "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


# get stages id in rank
def get_stages_in_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = None,
) -> tuple[int]:
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == None or style == 'loop':
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]
    raise ValueError(f"Unsupported schedule style: {style}")


def pipeline_model(
        layers,
        world_mesh,
        schedule_mode,
        n_microbatches = None,
        n_layers_per_stage = None,
        pp_dim="pp",
        loss_fn=None,
        losses = None,
        is_flatten_layers=True,
        **kwargs,
):
    if is_flatten_layers:
        layers = _flatten_layers(layers)
    pp_mesh = world_mesh[pp_dim]

    stages, model_chunks = _manual_split(layers, pp_mesh, schedule_mode, n_layers_per_stage, pp_dim, **kwargs)
    # TODO: build_pipeline_schedule
    pp_schedule = build_pipeline_schedule(stages, schedule_mode,n_microbatches,loss_fn,**kwargs)

    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_chunks, has_first_stage, has_last_stage



def _generate_splits(num_layers, pp_size, schedule_mode, n_layers_per_stage):
    # whether rank has one or more stages, num_layers should be divisible by pp_size
    assert num_layers % pp_size == 0, f"num_layers {num_layers} should be divisible by pp_size {pp_size}"

    is_single_stage_schedule = issubclass(schedule_mode, PipelineScheduleSingle)

    # Handle PipelineScheduleSingle
    if is_single_stage_schedule:
        num_stages_per_rank = 1
        assert n_layers_per_stage == None or num_layers / pp_size == n_layers_per_stage, \
            f"In PipelineScheduleSingle, n_layers_per_stage should be None or equals to num_layers/ pp_size {num_layers / pp_size}"
        n_layers_per_stage = num_layers // pp_size
    # Handle PipelineScheduleMulti
    else:
        if n_layers_per_stage is None:
            num_stages_per_rank = 2
            assert num_layers % (pp_size * num_stages_per_rank) == 0, \
                f"In PipelineScheduleMulti, when n_layers_per_stage is None, num_layers {num_layers} should be divisible by pp_size * num_stages_per_rank {pp_size} * {num_stages_per_rank}"
                
            n_layers_per_stage = num_layers // (pp_size * num_stages_per_rank)
        else:
            assert num_layers % (pp_size * n_layers_per_stage) == 0, \
                f"In PipelineScheduleMulti, when n_layers_per_stage is specified, num_layers {num_layers} should be divisible by pp_size * n_layers_per_stage {pp_size} * {n_layers_per_stage}"
            num_stages_per_rank = num_layers // (pp_size * n_layers_per_stage)
        

    total_num_stages = pp_size * num_stages_per_rank

    # split layers into {total_num_stages} stages, get the split spec

    splits = [i * n_layers_per_stage for i in range(1, total_num_stages)]

    return splits


def _manual_split(layers,pp_mesh,schedule_mode,n_layers_per_stage, pp_dim, **kwargs):
    pp_group = pp_mesh.get_group()
    pp_rank = pp_group.get_group_rank(dist.get_rank())
    pp_size = dist.get_world_size(pp_group)

    splits = _generate_splits(len(layers),pp_size, schedule_mode, n_layers_per_stage)


    def _build_stage(
        stage_idx: int,
        start_idx: int| None = None,
        stop_idx: int | None = None,
    ):
        def check_valid_index(layers, start_idx, stop_idx):
            if start_idx is None:
                start_idx = 0
            elif start_idx < 0 or start_idx > len(layers):
                    raise ValueError(f"Invalid index, start_idx: {start_idx} should be greater than zero and not greater than len(layers): {len(layers)}")
            
            if stop_idx is None:
                stop_idx = len(layers)
            elif stop_idx<0 or stop_idx > len(layers):
                    raise ValueError(f"Invalid index, stop_idx: {stop_idx} should be greater than zero and not greater than len(layers): {len(layers)}")
         
            if start_idx >= stop_idx:
                raise ValueError("Invalid index, stop_idx should be larger than start_idx")

            return start_idx, stop_idx

        start_idx, stop_idx = check_valid_index(layers, start_idx, stop_idx)

        #copy_layers = copy.deepcopy(layers[start_idx:stop_idx])

        sub_model = _Pipeline_model_chunk(layers[start_idx:stop_idx])

        stage = PipelineStage(sub_model, stage_idx, num_stages, group=pp_mesh.get_group(pp_dim))

        return stage, sub_model

    num_stages = len(splits) + 1

    # The rank may include more than one stage
    stage_idx = -1

    stages = []
    models = []


    style = _get_style(schedule_mode)
    for stage_idx in get_stages_in_rank(pp_rank, pp_size, num_stages, style=style):
        start_layer_idx = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer_idx = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer_idx,
            stop_layer_idx,
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models


def build_pipeline_schedule(stages, schedule_mode, n_microbatches, loss_fn,**kwargs):
    # calulate the number of microbatches
    batch_size = kwargs.get("batch_size", None)
    micro_batch_size = kwargs.get("micro_batch_size", None)

    if n_microbatches is None:
        if batch_size is not None and micro_batch_size is not None:
            assert batch_size % micro_batch_size == 0,f"batch_size {batch_size} should be divisible by micro_batch_size {micro_batch_size}"
            n_microbatches = batch_size // micro_batch_size
        else:
            n_microbatches = 1
    elif batch_size is not None and micro_batch_size is not None:
            assert n_microbatches == batch_size / micro_batch_size, f"n_microbatches {n_microbatches} should be equal to batch_size {batch_size} / micro_batch_size {micro_batch_size}"

    is_single_stage_schedule = issubclass(schedule_mode, PipelineScheduleSingle)

    schedule = schedule_mode(
        stages[0] if is_single_stage_schedule else stages,
        n_microbatches,
        loss_fn
    )
    return schedule
            

# use decorator factory to accept extra arguments
def pipeline(
        layers,
        world_mesh,
        schedule_mode,
        n_microbatches = None,
        n_layers_per_stage = None,
        pp_dim="pp",
        loss_fn=None,
        losses = None,
        is_flatten_layers=True,
        **kwargs,):

    """
    Decorator factory for pipeline model.
    """
    def decorator(cls): 
        original_init = cls.__init__


        def new_forward(*_args, **_kwargs):  # accept arbitrary arguments
            schedule, _, has_first_stage, has_last_stage = pipeline_model(
                layers,
                world_mesh,
                schedule_mode,
                n_microbatches,
                n_layers_per_stage,
                pp_dim,
                loss_fn,
                losses,
                is_flatten_layers,
                **kwargs,
            )
        
            labels = None
            if loss_fn is not None:
                assert "labels" in _kwargs, "labels is required for loss_fn"
                labels = _kwargs.pop("labels")

            if has_first_stage:
                schedule.step(*_args, **_kwargs)
            elif has_last_stage:
                schedule.step(target=labels,losses=losses,**_kwargs)
            else:
                schedule.step(**_kwargs) 


        # 计算 layer 所属的 ipp，并据此修改模型参数的分布式属性
        def assign_layers(self):
            nonlocal layers
            tmp_layers = []
            for layer_name in layers:
                assert hasattr(self, layer_name), f"layer: {layer_name} not in model"
                tmp_layers.append(getattr(self, layer_name))
            layers = tmp_layers

            if is_flatten_layers:
                layers = _flatten_layers(layers)
            
            style = _get_style(schedule_mode)

            # 计算 ipp 所拥有的layers
            pp_size = world_mesh[pp_dim].shape[0]

            splits = _generate_splits(len(layers),pp_size, schedule_mode, n_layers_per_stage)

            num_stages = len(splits) + 1

            for ipp in range(pp_size):
                stages = get_stages_in_rank(ipp, pp_size, num_stages, style)
                mesh_for_place = get_pp_mesh(ipp)
                for stage_idx in stages:
                    start_layer_idx = splits[stage_idx - 1] if stage_idx > 0 else 0
                    stop_layer_idx = splits[stage_idx] if stage_idx < num_stages - 1 else num_stages
                    while start_layer_idx < stop_layer_idx:
                        # shard_tensor
                        if hasattr(layers[start_layer_idx], 'weight'):
                            dist.shard_tensor(layers[start_layer_idx].weight, mesh_for_place, [Replicate()]*len(mesh_for_place.shape))
                        if hasattr(layers[start_layer_idx], 'bias'):
                            dist.shard_tensor(layers[start_layer_idx].bias, mesh_for_place, [Replicate()]*len(mesh_for_place.shape))
                        start_layer_idx += 1
        

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assign_layers(self)
            self.forward = new_forward

        cls.__init__ = new_init

        return cls

    return decorator
