#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import copy
import inspect

import paddle
from paddle.framework import Block
from paddle.static import Parameter, Variable

from .dist_attribute import TensorDistAttr
from .utils import __no_shape_var_type__, _linear_idx2coordinate


class DistributedTensor:
    """
    DistributedTensor represents the distribution of tensor on the process group and
    local tensors can be created by DistributedTensor.
    Only support even sharding now and uneven sharding will be supported in the future.
    Local tensor information can be obtained from the DistributedTensor instance object,
    or obtained by the static methods provided by DistributedTensor,
    including shard (i.e. the index in the serial tensor), offsets, and sizes.
    """

    @staticmethod
    def _validate_sizes_and_dist_attr(
        sizes, dims_mapping, topology, processes, rank=None, shard_sizes=None
    ):
        if not (
            isinstance(sizes, (list, tuple))
            and all(isinstance(x, int) and x >= 0 for x in sizes)
        ):
            raise ValueError(
                "The sizes must be list or tuple and item in sizes must be non-negative integer, but got {}".format(
                    sizes
                )
            )
        if not (
            isinstance(dims_mapping, (list, tuple))
            and all(isinstance(x, int) and x >= -1 for x in dims_mapping)
        ):
            raise ValueError(
                "The dims_mapping must be list or tuple and item in dims_mapping must >= -1, but got {}".format(
                    dims_mapping
                )
            )
        if not (
            isinstance(processes, (list, tuple))
            and all(isinstance(x, int) and x >= 0 for x in processes)
        ):
            raise ValueError(
                "The processes must be list or tuple and item in processes must be integer, but got {}".format(
                    processes
                )
            )
        if not (
            isinstance(topology, (list, tuple))
            and all(isinstance(x, int) and x > 0 for x in topology)
        ):
            raise ValueError(
                "The topology must be list or tuple and item in topology must be non-negative integer, but got {}".format(
                    topology
                )
            )
        if rank is not None and not (isinstance(rank, int) and rank >= 0):
            raise ValueError(f"The rank must >= 0, but got {rank}")

        # # NOTE: Only support even sharding now
        # if shard_sizes is not None:
        #     raise ValueError("Only support even sharding now.")

    @staticmethod
    def get_local_sizes(
        global_sizes,
        dims_mapping,
        topology,
        processes,
        rank=None,
        shard_sizes=None,
    ):
        DistributedTensor._validate_sizes_and_dist_attr(
            global_sizes, dims_mapping, topology, processes, rank, shard_sizes
        )

        local_sizes = []
        # for even sharding, the local sizes of every rank are equal

        for idx, item in enumerate(global_sizes):
            # This is a trick to avoid dims_mapping is []
            val = dims_mapping[idx] if idx < len(dims_mapping) else -1
            if val == -1:
                local_sizes.append(item)
            else:
                local_sizes.append(item // topology[dims_mapping[idx]])

        return local_sizes

    @staticmethod
    def get_local_offsets(
        global_sizes, dims_mapping, topology, processes, rank, shard_sizes=None
    ):
        local_sizes = DistributedTensor.get_local_sizes(
            global_sizes, dims_mapping, topology, processes, rank, shard_sizes
        )
        local_offsets = []
        rank_relatvie = processes.index(rank)
        coordinate = _linear_idx2coordinate(topology, rank_relatvie)

        for i in range(len(global_sizes)):
            if dims_mapping[i] == -1:
                local_offsets.append(0)
            else:
                local_offsets.append(
                    coordinate[dims_mapping[i]] * local_sizes[i]
                )
        return local_offsets

    @staticmethod
    def get_global_sizes(
        local_sizes,
        dims_mapping,
        topology,
        processes,
        rank=None,
        shard_sizes=None,
    ):
        DistributedTensor._validate_sizes_and_dist_attr(
            local_sizes, dims_mapping, topology, processes, rank, shard_sizes
        )
        global_sizes = []
        for idx, item in enumerate(local_sizes):
            if dims_mapping[idx] == -1:
                global_sizes.append(item)
            else:
                global_sizes.append(item * topology[dims_mapping[idx]])
        return global_sizes

    @staticmethod
    def get_local_shard(
        global_sizes, dims_mapping, topology, processes, rank, shard_sizes=None
    ):
        local_offsets = DistributedTensor.get_local_offsets(
            global_sizes, dims_mapping, topology, processes, rank, shard_sizes
        )
        local_sizes = DistributedTensor.get_local_sizes(
            global_sizes, dims_mapping, topology, processes, rank, shard_sizes
        )
        assert len(local_sizes) == len(
            local_offsets
        ), "The length of local_sizes must be equal to local_offsets, but got {} and {}.".format(
            len(local_sizes), len(local_offsets)
        )

        local_end_offsets = [
            x[0] + x[1] for x in zip(local_offsets, local_sizes)
        ]
        local_shard = list(zip(local_offsets, local_end_offsets))
        return local_shard

    def __init__(self, serial_tensor, dist_attr=None, dist_context=None):
        self._serial_tensor = serial_tensor
        if dist_attr is not None and isinstance(dist_attr, TensorDistAttr):
            # TODO: remove this deepcopy after we fix the issue
            self._dist_attr = copy.deepcopy(dist_attr)
            # self._dist_attr = dist_attr
            # TODO: Do we really need to write dist_attr back to serial_tensor？
            self._serial_tensor.dist_attr = dist_attr
        else:
            assert dist_attr is None, f"{dist_attr}"
            # Use the dist attr of serial_tensor to do the initialization
            self._dist_attr = self._serial_tensor.dist_attr

        self._batch_dim = 0
        self._local_offsets_map = {}
        self._local_shard_map = {}
        self._local_tensor_map = {}

        from .dist_context import get_default_distributed_context

        self._dist_context = (
            dist_context
            if dist_context is not None
            else get_default_distributed_context()
        )
        # TODO: Add Automatically to dist_context after initialized and it will be adapted in the future.
        # self._dist_context.add_dist_tensor_for_program(self)

    @property
    def serial_tensor(self):
        return self._serial_tensor

    @property
    def dist_attr(self):
        return self._dist_attr

    @dist_attr.setter
    def dist_attr(self, dist_attr):
        self._dist_attr = dist_attr
        # TODO: Do we really need to write back dist_attr to serial_tensor？
        self._serial_tensor.dist_attr = dist_attr

    @property
    def dist_context(self):
        return self._dist_context

    # def _init_default_dist_attr(self):
    #     if self._dist_attr.dims_mapping is None:
    #         if self.serial_tensor.type in __no_shape_var_type__:
    #             tensor_shape = []
    #         else:
    #             tensor_shape = self._serial_tensor.shape
    #         tensor_dims_mapping = [-1 for _ in range(len(tensor_shape))]
    #         self._dist_attr.dims_mapping = tensor_dims_mapping

    def validate_dist_attr(self):
        if self.serial_tensor.type in __no_shape_var_type__:
            return True
        tensor_shape = self.serial_tensor.shape
        if len(tensor_shape) != len(self.dist_attr.dims_mapping):
            return False
        for i in range(len(self.dist_attr.dims_mapping)):
            if self.dist_attr.dims_mapping[
                i
            ] < -1 or self.dist_attr.dims_mapping[i] >= len(
                self.dist_attr.process_mesh.shape
            ):
                return False
        for i in range(len(self.dist_attr.process_mesh.shape)):
            if self.dist_attr.dims_mapping.count(i) > 1:
                return False
        return True

    def local_sizes(self, rank=None):
        """Get local sizes of the given rank."""
        rank = paddle.distributed.get_rank() if rank is None else rank
        global_sizes = self.serial_tensor.shape
        dims_mapping = self.dist_attr.dims_mapping
        # shard_sizes = self.dist_attr.shard_sizes
        processes = self.dist_attr.process_mesh.process_ids
        topology = self.dist_attr.process_mesh.shape
        local_sizes = DistributedTensor.get_local_sizes(
            global_sizes, dims_mapping, topology, processes, rank
        )

        return local_sizes

    def local_offsets(self, rank=None):
        rank = paddle.distributed.get_rank() if rank is None else rank
        local_offsets = None
        if rank in self._local_offsets_map.keys():
            local_offsets = self._local_offsets_map[rank]
        else:
            global_sizes = self.serial_tensor.shape
            dims_mapping = self.dist_attr.dims_mapping
            # shard_sizes = self.dist_attr.shard_sizes
            processes = self.dist_attr.process_mesh.process_ids
            topology = self.dist_attr.process_mesh.shape
            local_offsets = DistributedTensor.get_local_offsets(
                global_sizes, dims_mapping, topology, processes, rank
            )
            self._local_offsets_map[rank] = local_offsets

        return local_offsets

    def global_sizes(self):
        return self.serial_tensor.shape

    def local_shard(self, rank=None):
        rank = paddle.distributed.get_rank() if rank is None else rank
        local_shard = None
        if rank in self._local_shard_map.keys():
            local_shard = self._local_shard_map[rank]
        else:
            global_sizes = self.serial_tensor.shape
            dims_mapping = self.dist_attr.dims_mapping
            # shard_sizes = self.dist_attr.shard_sizes
            processes = self.dist_attr.process_mesh.process_ids
            topology = self.dist_attr.process_mesh.shape
            local_shard = DistributedTensor.get_local_shard(
                global_sizes, dims_mapping, topology, processes, rank
            )
            self._local_shard_map[rank] = local_shard

        return local_shard

    def new_local_tensor(self, block=None, rank=None, name=None):
        """
        Create a new local tensor of serial tensor corresponding to rank.
        Args:
            block (Block): The block contains the new tensor. Default value is recommend and it will be created in the block of dist main program corresponding to the serial tensor block id. Default: None.
            rank (int): The rank id. Default value is recommend and it will be the current rank. Default: None.
        """

        def _copy_kwargs(serial_tensor):
            kwargs = {}
            no_need_copy_args = ["self", "block", "shape", "name"]
            arg_spec = inspect.getfullargspec(Variable.__init__)

            for key in arg_spec.args:
                # TODO: Check the copied attribute from serial tensor whether valid
                if key in no_need_copy_args:
                    continue
                elif key not in kwargs:
                    if key == "type":
                        kwargs[key] = serial_tensor.desc.type()
                    elif key == "dtype":
                        kwargs[key] = serial_tensor.desc.dtype()
                    elif key == "lod_level":
                        kwargs[key] = serial_tensor.desc.lod_level()
                    elif key == "persistable":
                        kwargs[key] = serial_tensor.desc.persistable()
                    elif key == "stop_gradient":
                        kwargs[key] = serial_tensor.desc.stop_gradient()
                    elif key == "need_check_feed":
                        kwargs[key] = serial_tensor.desc.need_check_feed()
                    # TODO: Get capacity by framework
                    elif key == "capacity":
                        continue
                    else:
                        kwargs[key] = self.serial_tensor.__dict__[key]

            if isinstance(serial_tensor, Parameter):
                kwargs["trainable"] = serial_tensor.trainable
                kwargs["optimize_attr"] = serial_tensor.trainable
                kwargs["regularizer"] = serial_tensor.regularizer
                kwargs["do_model_average"] = serial_tensor.do_model_average
                kwargs["need_clip"] = serial_tensor.need_clip
                kwargs["is_distributed"] = serial_tensor.is_distributed
                kwargs["is_parameter"] = serial_tensor.is_parameter

            return kwargs

        if rank is not None and not (isinstance(rank, int) and rank >= 0):
            raise ValueError(f"The rank must >= 0, but got {rank}")
        if block is not None and not isinstance(block, Block):
            raise TypeError(f"The block must be Block, but got {type(block)}.")
        rank = paddle.distributed.get_rank() if rank is None else rank

        if block is None:
            block_id = self.serial_tensor.block.idx
            block = self.dist_context.dist_main_programs[rank].block(block_id)

        # copy serial tensor attribute
        kwargs = _copy_kwargs(self.serial_tensor)
        kwargs["name"] = name
        kwargs["shape"] = self.local_sizes(rank)

        if isinstance(self.serial_tensor, Parameter):
            kwargs.pop("persistable")
            local_tensor = Parameter(block=block, **kwargs)
        else:
            local_tensor = block.create_var(**kwargs)

        # TODO: Set original id when set original_id is approved
        local_tensor.desc.set_original_id(self.serial_tensor.desc.id())
        self._local_tensor_map[rank] = local_tensor
        return local_tensor

    def local_tensor(self, rank=None):
        rank = paddle.distributed.get_rank() if rank is None else rank
        assert (
            rank in self._local_tensor_map
        ), f"The rank {rank} local tensor has not been created."
        return self._local_tensor_map[rank]

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_serial_tensor" or k == "_local_tensor_map":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __str__(self):
        str = "{{tensor name: {}, tensor id: {}, tensor original_id {}".format(
            self.serial_tensor.desc.name(),
            self.serial_tensor.desc.id(),
            self.serial_tensor.desc.original_id(),
        )

        # str += ", {}".format(self.dist_attr)
        # return str

        if self.dist_attr.is_annotated("process_mesh"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", process_mesh ({}): {}".format(
            annotated_str, self.dist_attr.process_mesh
        )

        str += f", is_parameter: {self.serial_tensor.is_parameter}"

        if self.dist_attr.is_annotated("dims_mapping"):
            annotated_str = "annotated"
        else:
            annotated_str = "non-annotated"
        str += ", dims_mapping ({}): {} }}".format(
            annotated_str, self.dist_attr.dims_mapping
        )

        # if self.dist_attr.is_annotated("shard_mask"):
        #     annotated_str = "annotated"
        # else:
        #     annotated_str = "non-annotated"
        # str += ", shard_mask ({}): {}".format(annotated_str, None)

        # if self.dist_attr.is_annotated("offload_device"):
        #     annotated_str = "annotated"
        # else:
        #     annotated_str = "non-annotated"
        # str += ", offload_device ({}): {} }}".format(annotated_str, None)
        return str
