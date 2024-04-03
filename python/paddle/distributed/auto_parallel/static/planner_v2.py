# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import os
import pickle
import sys

import numpy as np

from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.static.dist_attribute import (
    OperatorDistAttr,
    TensorDistAttr,
)
from paddle.distributed.auto_parallel.static.dist_op import DistributedOperator
from paddle.distributed.auto_parallel.static.dist_tensor import (
    DistributedTensor,
)

from ...utils.log_utils import get_logger
from .completion import Completer
from .dist_context import get_default_distributed_context
from .tuner.parallel_tuner import ParallelTuner
from .tuner.rule_based_tuner import RuleBasedTuner
from .utils import is_naive_data_parallel


class Planner:
    def __init__(self, mode, dist_context):
        self._mode = mode
        self._dist_context = dist_context
        self._load = False  # load dist_attr from file

        # NOTE: [HighOrderGrad]. There are grad ops in forward phase, and it need
        # dependency of backward-forward ops in forward completion.
        default_ctx = get_default_distributed_context()
        self._dist_context._dist_op_context = default_ctx.dist_op_context
        self._dist_context.data_parallel = default_ctx.data_parallel
        if not is_naive_data_parallel(self._dist_context):
            # Use SSA graph for complex parallelism
            self._dist_context.initialize(with_graph=True)
        else:
            # Use program for data parallel parallelism
            self._dist_context.initialize(with_graph=False)

        self._completer = Completer(self._dist_context)

        self._strategy = dist_context.strategy
        # set parallel tuner for auto search
        if self._strategy.auto_mode == "full_random":
            self._parallel_tuner = ParallelTuner(
                self._dist_context, mode=self._mode
            )
        elif self._strategy.auto_mode == "full_rule_based":
            self._parallel_tuner = RuleBasedTuner(
                self._dist_context, mode=self._mode
            )

    @property
    def completer(self):
        return self._completer

    def plan(self):
        logger = get_logger(logging.INFO)
        path = None
        if self._dist_context._json_config:
            try:
                path = self._dist_context._json_config["tuner_load_path"]
            except:
                path = None
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    dist_attrs = pickle.load(f)
                tensor_dist_attrs = dist_attrs["tensor"]
                op_dist_attrs = dist_attrs["op"]
                process_meshes = dist_attrs["process_meshes"]
                cluster = dist_attrs["cluster"]
                last_gpu_model = cluster.machines[0].devices[0].model
                last_gpu_memory = cluster.machines[0].devices[0].memory
                last_node_count = len(cluster.machines)
                last_device_count = len(cluster.get_all_devices("GPU"))

                gpu_model = (
                    self._dist_context.cluster.machines[0].devices[0].model
                )
                gpu_memory = (
                    self._dist_context.cluster.machines[0].devices[0].memory
                )
                node_count = len(self._dist_context.cluster.machines)
                device_count = len(
                    self._dist_context.cluster.get_all_devices("GPU")
                )
                if (
                    gpu_model != last_gpu_model
                    or gpu_memory != last_gpu_memory
                    or last_node_count != node_count
                    or device_count != last_device_count
                ):
                    logger.info(
                        f"The cluster {node_count} nodes {device_count} {gpu_model} devices is different from the saved last cluster {last_node_count} nodes {last_device_count} {last_gpu_model} devices, so we run the planner again."
                    )
                    need_set_dist_attr = False
                else:
                    need_set_dist_attr = True
            except:
                need_set_dist_attr = False

            if need_set_dist_attr:
                for key in op_dist_attrs:
                    serial_op = self._dist_context._dist_ops_for_program[
                        key
                    ].serial_op
                    # clear dist attr
                    serial_op.dist_attr = OperatorDistAttr(serial_op.desc)
                    serial_op.dist_attr.parse_from_string(op_dist_attrs[key])
                    self._dist_context._dist_ops_for_program[
                        key
                    ] = DistributedOperator(serial_op)

                for key in tensor_dist_attrs:
                    serial_tensor = (
                        self._dist_context._dist_tensors_for_program[
                            key
                        ].serial_tensor
                    )
                    # clear dist attr
                    serial_tensor.dist_attr = TensorDistAttr(serial_tensor.desc)
                    serial_tensor.dist_attr.parse_from_string(
                        tensor_dist_attrs[key]
                    )
                    self._dist_context._dist_tensors_for_program[
                        key
                    ] = DistributedTensor(serial_tensor)

                process_meshes = []
                for item in dist_attrs["process_meshes"]:
                    process_ids = item[0]
                    shape = item[1]
                    process_meshes.append(
                        ProcessMesh(
                            np.array(process_ids).reshape(shape).tolist()
                        )
                    )

                self._dist_context.process_meshes = process_meshes
                self._load = True

                logger.info(
                    f"The parallel strategy has been loaded from {path}"
                )

        if not self._load:
            if self._strategy.auto_mode != "semi":
                self._parallel_tuner.tune()
            else:
                self._completer.complete_forward_annotation()

        if os.getenv("PADDLE_AUTO_PARALLEL_STAGE", "run") != "run":
            sys.exit()

        # parse forward sub block
        self._dist_context.block_state.parse_forward_blocks(
            self._dist_context.serial_main_program
        )
