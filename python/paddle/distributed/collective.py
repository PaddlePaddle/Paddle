#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import pickle
import io
import datetime
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import in_dygraph_mode
from ..fluid.framework import _non_static_mode
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.layers.tensor import fill_constant
import paddle
import paddle.fluid.core as core
from paddle import _legacy_C_ops
from .fleet.layers.mpu.mp_ops import split  # noqa: F401
from .fleet.layers.mpu.mp_ops import _c_identity  # noqa: F401
from .fleet.layers.mpu.mp_ops import _c_concat  # noqa: F401
from .fleet.layers.mpu.mp_ops import _c_split  # noqa: F401
from .fleet.layers.mpu.mp_ops import _mp_allreduce  # noqa: F401
from .fleet.layers.mpu.mp_ops import _c_lookup_table  # noqa: F401
from .fleet.layers.mpu.mp_ops import _Linear  # noqa: F401
from .fleet.layers.mpu.mp_ops import _set_var_distributed  # noqa: F401
from .fleet.layers.mpu.mp_ops import _c_softmax_with_cross_entropy  # noqa: F401
from .fleet.layers.mpu.mp_ops import _linear  # noqa: F401
from .fleet.layers.mpu.mp_ops import _parallel_linear  # noqa: F401
from .fleet.layers.mpu.mp_ops import _parallel_embedding  # noqa: F401
from .communication.group import Group, _add_new_group, is_initialized

__all__ = []

_global_env = None


def _get_global_env():
    global _global_env
    if not _global_env:
        _global_env = paddle.distributed.ParallelEnv()
    return _global_env


# group map : the map of all group, 0 for GlobalGroup
# Dict[int, Group]
_group_map = {}
_global_env_gid = 0

# group map by name : the map of all groups from their names
# Dict[name, Group]
_group_map_by_name = {}

# backend map by group : the map of all backend from their groups
# Dict[group, backend]
_group_map_backend = {}

# Name of the default group for init_parallel_env
_default_group_name = "_default_pg"

_valid_backend_list = ['nccl', 'gloo', 'hccl', 'heter', 'xccl', 'bkcl']
_default_store = None  # the default tcp store
_default_backend = None
_default_timeout = datetime.timedelta(seconds=1800)
_start_ring_id = 0


def _set_default_backend(backend):
    global _default_backend
    _default_backend = backend


def _set_default_store(store):
    global _default_store
    _default_store = store


def _get_group_map():
    global _group_map
    if _global_env_gid not in _group_map:
        genv = _get_global_env()
        _group_map[_global_env_gid] = Group(
            genv.rank, 0, list(range(genv.world_size))
        )
    return _group_map


def _get_global_group():
    return _get_group_map()[_global_env_gid]


def _get_group_map_by_name():
    global _group_map_by_name
    return _group_map_by_name


def _get_default_group():
    global _group_map_by_name
    assert is_initialized(), (
        "Call paddle.distributed.init_parallel_env first "
        "to initialize the distributed environment."
    )
    return _get_group_map_by_name()[_default_group_name]


def _set_group_map(gid, group):
    global _group_map
    assert gid not in _group_map
    _group_map[gid] = group


def _set_group_map_by_name(name, group):
    global _group_map_by_name
    assert name not in _group_map_by_name
    _group_map_by_name[name] = group


def _set_group_map_backend(group, backend):
    global _group_map_backend
    assert group not in _group_map_backend
    _group_map_backend[group] = backend


def _new_ring_id():
    # NOTE(liyurui): For compatible reason, auto parallel and eager mode relay on previous syntax.
    if in_dygraph_mode():
        global _start_ring_id
        _start_ring_id += 1
        return _start_ring_id + max(_get_global_env().nrings, 9)
    else:
        return len(_get_group_map()) + max(_get_global_env().nrings, 9)


def _new_process_group_impl(
    backend,
    store,
    rank,
    world_size,
    group_name,
    pg_options,
    group_id=0,
):
    pg = None
    genv = _get_global_env()
    assert backend in _valid_backend_list, "Unsupported backend: %s." % backend
    if backend == "gloo":
        pg = core.ProcessGroupGloo(store, rank, world_size, group_id)
    elif backend == "nccl":
        pg = core.ProcessGroupNCCL(store, rank, world_size, group_id)
    elif backend == "xccl":
        pg = core.ProcessGroupCustom(
            store, genv.device_type, rank, world_size, group_id
        )
    elif backend == "bkcl":
        pg = core.ProcessGroupBKCL(store, rank, world_size, group_id)
    return pg


def barrier(group=None):
    """

    Barrier among all participators in the group.

    Args:
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed import init_parallel_env

            paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
            init_parallel_env()
            paddle.distributed.barrier()
    """
    if group is not None and not group.is_member():
        return

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        place = paddle.fluid.framework._current_expected_place()
        if isinstance(place, paddle.fluid.core.CPUPlace):
            task = group.process_group.barrier()
        else:
            device_id = place.get_device_id()
            task = group.process_group.barrier(device_id)
        task.wait()
        return

    ring_id = 0 if group is None else group.id

    temp = fill_constant([1], dtype="int32", value="1")
    if _non_static_mode():
        return _legacy_C_ops.barrier(temp, temp, 'ring_id', ring_id)

    op_type = 'barrier'

    if not isinstance(ring_id, int):
        raise ValueError("The type of 'group' for barrier must be int.")
    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [temp]},
        outputs={'Out': [temp]},
        attrs={'ring_id': ring_id},
    )


# _custom_gid provides a way for users to
# set the group id, which is usually useful
# to be compatible with the static mode.
_custom_gid = None


def _set_custom_gid(gid):
    global _custom_gid
    _custom_gid = gid


def new_group(ranks=None, backend=None, timeout=_default_timeout):
    """

    Creates a new distributed communication group.

    Args:
        ranks (list): The global ranks of group members.
        backend (str): The backend used to create group, only nccl is supported now.
        timeout (datetime.timedelta, optional): The waiting timeout for store relevant options, default is 30 minutes.

    Returns:
        Group: The group instance.

    Examples:
        .. code-block:: python

            import paddle

            paddle.distributed.init_parallel_env()
            tindata = paddle.randn(shape=[2, 3])
            gp = paddle.distributed.new_group([2,4,6])
            paddle.distributed.all_reduce(tindata, group=gp, sync_op=False)

    """
    global _custom_gid
    global _group_map
    if in_dygraph_mode():
        global _default_group_name
        gid = _custom_gid if _custom_gid else _new_ring_id()
        group_name = _default_group_name + str(gid)
        if backend != 'heter' and (ranks is None or len(ranks) > 1):
            global_group = _get_default_group()
            global_rank = global_group.rank
            global_ranks = global_group.ranks
            backend = _default_backend if backend is None else backend
            if ranks is None:
                ranks = global_ranks
            assert len(ranks) <= len(global_ranks), (
                "Size of new group must be less than or "
                "equal to that of the default global group."
            )
        size = len(ranks)
        ranks = sorted(ranks)
        if size > 1 and global_rank in ranks:
            rank = 0 if backend == 'heter' else ranks.index(global_rank)
            pg = _new_process_group_impl(
                backend,
                _default_store,
                rank,
                size,
                group_name,
                pg_options=None,
                group_id=gid,
            )
        else:
            rank = -1
            pg = None
        group = Group(rank, gid, ranks, pg=pg, name=group_name)
        _group_map_by_name[group_name] = group
        _group_map[gid] = group
        _group_map_backend[group] = backend
        # TODO: The method below is a new method for group management, will replace the previous
        # three in the future.
        _add_new_group(group)

        # TODO(shenliang03): This is a temporary solution to solve the problem of
        # hang caused by tcp
        paddle.distributed.barrier(group=group)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()
        return group

    if not backend:
        backend = 'nccl'
    assert backend == 'nccl', "backend other than nccl is not supported yet"

    genv = _get_global_env()
    global_rank = genv.rank

    ring_id = _new_ring_id()

    if global_rank not in ranks:
        gp = Group(-1, ring_id, ranks)
        _group_map[ring_id] = gp
    else:
        ranks = sorted(ranks)
        group_rank = ranks.index(global_rank)
        group_size = len(ranks)
        gp = Group(group_rank, ring_id, ranks)
        _group_map[ring_id] = gp

        if group_size >= 2:
            strategy = core.ParallelStrategy()
            strategy.nranks = group_size
            strategy.local_rank = group_rank
            strategy.trainer_endpoints = [
                genv.trainer_endpoints[i] for i in ranks
            ]
            strategy.current_endpoint = genv.current_endpoint
            strategy.nrings = 1

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(genv.device_id)
                core.NCCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            elif core.is_compiled_with_npu():
                place = core.NPUPlace(genv.device_id)
                core.HCCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            elif core.is_compiled_with_mlu():
                place = core.MLUPlace(genv.device_id)
                core.CNCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            elif core.is_compiled_with_xpu():
                place = core.XPUPlace(genv.device_id)
                core.BKCLParallelContext(strategy, place).init_with_ring_id(
                    ring_id
                )
            else:
                assert False, "no cuda device found"
        else:
            return gp

    # TODO(shenliang03): This is a temporary solution to solve the problem of
    # hang caused by cross-creation of new_group
    tmp = (
        paddle.to_tensor([1], dtype="int32")
        if _non_static_mode()
        else fill_constant([0], dtype="int32", value="1")
    )
    paddle.distributed.all_reduce(tmp, sync_op=True)
    paddle.distributed.wait(tmp)
    return gp


def wait(tensor, group=None, use_calc_stream=True):
    """

    wait to sync stream for group.

    Args:
        tensor (Tensor): The Tensor used before sync.
        group (Group): The Group instance to perform sync.
        use_calc_stream (bool): Wether to use calculation stream (True) or communication stream (False).
            Default to True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle

            paddle.distributed.init_parallel_env()
            tindata = paddle.randn(shape=[2, 3])
            paddle.distributed.all_reduce(tindata, sync_op=True)
            paddle.distributed.wait(tindata)

    """

    if group is not None and not group.is_member():
        return

    ring_id = 0 if group is None else group.id

    if use_calc_stream:
        _sync_calc_stream(tensor)
    else:
        _sync_comm_stream(tensor, ring_id)


def _sync_calc_stream(tensor):

    if _non_static_mode():
        return _legacy_C_ops.c_sync_calc_stream(tensor, tensor)

    op_type = 'c_sync_calc_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
    )


def _sync_comm_stream(tensor, ring_id=0):

    if _non_static_mode():
        return _legacy_C_ops.c_sync_comm_stream(
            [tensor], [tensor], 'ring_id', ring_id
        )

    op_type = 'c_sync_comm_stream'

    helper = LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'X': [tensor]},
        outputs={'Out': [tensor]},
        attrs={'ring_id': ring_id},
    )


def all_gather(tensor_list, tensor, group=None, sync_op=True):
    """

    Gather tensors from all participators and all get the result. As shown
    below, one process is started with a GPU and the data of this process is represented
    by its group rank. Through the all_gather operator, each GPU will have data
    from all GPUs.

    .. image:: https://githubraw.cdn.bcebos.com/PaddlePaddle/docs/develop/docs/api/paddle/distributed/img/allgather.png
        :width: 800
        :alt: all_gather
        :align: center

    Args:
        tensor_list (list): A list of output Tensors. Every element in the list must be a Tensor whose data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool, bfloat16, complex64 or complex128.
        tensor (Tensor): The Tensor to send. Its data type
            should be float16, float32, float64, int32, int64, int8, uint8, bool, complex64 or complex128.
        group (Group, optional): The group instance return by new_group or None for global default group.
        sync_op (bool, optional): Whether this op is a sync op. The default value is True.

    Returns:
        None.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            tensor_list = []
            if dist.get_rank() == 0:
                data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
            else:
                data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
            dist.all_gather(tensor_list, data)
            print(tensor_list)
            # [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)
    """
    if group is not None and not group.is_member():
        return

    def convert_to_complex(list_of_tensor):
        list_of_complex = []
        for tensor in list_of_tensor:
            list_of_complex.append(paddle.as_complex(tensor))
        return list_of_complex

    is_input_complex = (
        tensor.dtype == paddle.complex64 or tensor.dtype == paddle.complex128
    )
    if is_input_complex:
        tensor = paddle.as_real(tensor)

    if in_dygraph_mode():
        group = _get_default_group() if group is None else group
        if len(tensor_list) == 0:
            tensor_shape = list(tensor.shape)
            tensor_shape[0] *= group.nranks
            out = paddle.empty(tensor_shape, tensor.dtype)
        else:
            out = paddle.concat(tensor_list, axis=0)
        task = group.process_group.all_gather_into_tensor(out, tensor, sync_op)
        task.wait()
        tensor_list.clear()
        list_of_tensor = paddle.split(out, group.nranks, 0)
        if is_input_complex:
            tensor_list.extend(convert_to_complex(list_of_tensor))
        else:
            tensor_list.extend(list_of_tensor)
        return

    use_calc_stream = sync_op
    ring_id = 0 if group is None else group.id
    nranks = _get_global_group().nranks if group is None else group.nranks

    if _non_static_mode():
        out = _legacy_C_ops.c_allgather(
            tensor,
            'use_calc_stream',
            use_calc_stream,
            'ring_id',
            ring_id,
            'nranks',
            nranks,
        )
    else:
        op_type = 'c_allgather'
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(dtype=tensor.dtype)
        if not isinstance(tensor_list, list):
            raise ValueError(
                "The type of 'tensor_list' for all_gather " "should be list."
            )
        for elem in tensor_list:
            check_variable_and_dtype(
                elem,
                'tensor_list',
                [
                    'float16',
                    'float32',
                    'float64',
                    'int32',
                    'int64',
                    'bool',
                    'int8',
                    'uint8',
                    'complex64',
                    'complex128',
                ],
                'all_gather',
            )
        check_variable_and_dtype(
            tensor,
            'tensor',
            [
                'float16',
                'float32',
                'float64',
                'int32',
                'int64',
                'bool',
                'int8',
                'uint8',
                'complex64',
                'complex128',
            ],
            'all_gather',
        )
        helper.append_op(
            type=op_type,
            inputs={'X': [tensor]},
            outputs={'Out': [out]},
            attrs={
                'ring_id': ring_id,
                'use_calc_stream': use_calc_stream,
                'nranks': nranks,
            },
        )

    list_of_tensor = paddle.split(out, nranks, 0)
    if is_input_complex:
        tensor_list.extend(convert_to_complex(list_of_tensor))
    else:
        tensor_list.extend(list_of_tensor)


def _convert_object_to_tensor(obj):
    _pickler = pickle.Pickler
    f = io.BytesIO()
    _pickler(f).dump(obj)
    data = np.frombuffer(f.getvalue(), dtype=np.uint8)
    tensor = paddle.to_tensor(data)
    return tensor, tensor.numel()


def _convert_tensor_to_object(tensor, len_of_tensor):
    _unpickler = pickle.Unpickler
    return _unpickler(io.BytesIO(tensor.numpy()[:len_of_tensor])).load()


def all_gather_object(object_list, obj, group=None):
    """

    Gather picklable objects from all participators and all get the result. Similiar to all_gather(), but python object can be passed in.

    Args:
        object_list (list): A list of output object. The datatype of every element in the list is same as the input obj.
        obj (Any): The picklable object to send.
        group (Group): The group instance return by new_group or None for global default group.

    Returns:
        None.

    Warning:
        This API only supports the dygraph mode.

    Examples:
        .. code-block:: python

            # required: distributed
            import paddle
            import paddle.distributed as dist

            dist.init_parallel_env()
            object_list = []
            if dist.get_rank() == 0:
                obj = {"foo": [1, 2, 3]}
            else:
                obj = {"bar": [4, 5, 6]}
            dist.all_gather_object(object_list, obj)
            print(object_list)
            # [{'foo': [1, 2, 3]}, {'bar': [4, 5, 6]}] (2 GPUs)
    """
    assert (
        in_dygraph_mode()
    ), "all_gather_object doesn't support static graph mode."

    tensor, len_of_tensor = _convert_object_to_tensor(obj)

    # gather len_of_tensor from all ranks
    list_len_of_tensor = []
    all_gather(list_len_of_tensor, len_of_tensor, group)
    # get the max length from list
    max_len_of_tensor = int(max(list_len_of_tensor).item())
    # resize the input tensor to max length avoid hang in all gather
    # Note(liyurui): Maybe we should support various length all_gather?
    # Now this operation is efficient for we don't support resize in python.
    numpy_data = tensor.numpy()
    numpy_data = np.resize(numpy_data, [max_len_of_tensor])
    input_tensor = paddle.to_tensor(numpy_data)

    tensor_list = []
    all_gather(tensor_list, input_tensor, group)
    for i, tensor in enumerate(tensor_list):
        object_list.append(
            _convert_tensor_to_object(tensor, list_len_of_tensor[i])
        )
