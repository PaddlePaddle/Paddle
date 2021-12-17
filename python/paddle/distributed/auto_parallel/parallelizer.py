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
# limitations under the License.

import os
import sys
import json
import shlex
import copy
import pathlib
import subprocess
import logging
import paddle
from paddle.distributed.utils import get_logger
from paddle.distributed.fleet import cloud_utils
import paddle.fluid.core as core
from .dist_context import DistributedContext
from .dist_context import get_default_distributed_context
from .dist_context import set_default_distributed_context
from .completion import complete_annotation, complete_backward_annotation
from .partitioner import Partitioner
from .process_group import get_all_process_groups
from .process_group import get_world_process_groups
from .utils import make_data_unshard
from .utils import set_grad_var_shape
from .reshard import reshard
from .cluster import Cluster
from .mapper import mapping
# from .auto_search import auto_search

_logger = get_logger(logging.INFO)


class AutoParallelizer:
    """
    AutoParallelizer is the main controller class to do the auto parallel process.
    And the auto parallel process will be triggered in the wrapped parallelize function.
    To facilitate the auto parallelization, it will contain information about program, cluster and the
    related context. In this basic version, the program information will be retrevied from 
    Fleet object, and the cluster information can be retrevied in the new created Cluster object,
    and the context information can be retrevied in the new created DistributedContext. 
    """

    def __init__(self, fleet):
        self._fleet = fleet
        self._optimizer = self._fleet.user_defined_optimizer
        self._dist_strategy = self._fleet._user_defined_strategy
        self._dist_context = DistributedContext()
        self._cluster = None
        self._cluster_topo_path = os.getenv("PADDLE_CLUSTER_TOPO_PATH", None)
        if self._cluster_topo_path is not None:
            self._cluster = Cluster()
            self._cluster.build_from_file(self._cluster_topo_path)
        # Prepare information for auto mapping
        self._rank_mapping_path = os.getenv("PADDLE_RANK_MAPPING_PATH", None)
        enable_auto_mapping_env = os.getenv("PADDLE_ENABLE_AUTO_MAPPING", None)
        if enable_auto_mapping_env is None:
            self._enable_auto_mapping = False
        else:
            self._enable_auto_mapping = True
        self._need_rank_mapping = os.getenv("PADDLE_NEED_RANK_MAPPING")
        self._need_rank_mapping = True if self._need_rank_mapping and \
            self._need_rank_mapping.lower() == 'true' else False

    def _remove_distributed_attrs(self, main_program):
        suffix = core.kAutoParallelSuffix()
        # distributed attributes for variable have been removed
        # in previous process.
        for block in main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    if suffix in attr_name:
                        op._remove_attr(attr_name)

    def _get_dist_program(self, dist_context, rank):
        # Annotation completion
        completed_main_program = complete_annotation(self._main_program,
                                                     dist_context)
        # Logical partition
        partitioner = Partitioner(self._dist_strategy, dist_context, rank)
        dist_main_prog, dist_startup_prog = partitioner.transpile_forward(
            completed_main_program, self._startup_program)
        dist_params_grads = partitioner.apply_backward(
            self._loss, completed_main_program, self._startup_program,
            dist_main_prog, dist_startup_prog)
        dist_optimize_ops = partitioner.apply_optimize(
            copy.deepcopy(self._optimizer), dist_params_grads, dist_main_prog,
            dist_startup_prog)

        make_data_unshard(dist_main_prog, dist_startup_prog, dist_context)

        reshard(dist_main_prog, dist_startup_prog, rank, dist_context)

        return dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog

    def parallelize(self,
                    loss,
                    startup_program,
                    parameter_list=None,
                    no_grad_set=None):
        assert startup_program is not None
        self._loss = loss
        self._startup_program = startup_program
        self._main_program = loss.block.program
        self._parameter_list = parameter_list
        self._no_grad_set = no_grad_set

        if self._enable_auto_mapping and self._need_rank_mapping:
            # Do the mapping pass before parallelization
            assert self._cluster is not None, \
                "The cluster must not be none when using auto mapping."
            dist_programs = {}
            world_process_group = get_world_process_groups()
            for rank in world_process_group.ranks:
                dist_context = DistributedContext()
                dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog = self._get_dist_program(
                    dist_context, rank)
                dist_programs[rank] = dist_main_prog

            # Do the mapping between the distributed program graph and the cluster graph
            rank_mapping_dict = mapping(dist_programs, self._cluster)
            rank_mapping = list(rank_mapping_dict.values())

            # Relaunch the training by using the rank mapping file
            with open(self._rank_mapping_path, "w") as rank_mapping_file:
                json.dump(rank_mapping, rank_mapping_file)

            enable_elastic = os.getenv("PADDLE_ENABLE_ELASTIC")
            enable_elastic = True if enable_elastic and enable_elastic.lower(
            ) == 'true' else False
            if enable_elastic:
                print("Auto mapping finished, now do elastic re-launch")
                sys.exit(paddle.distributed.fleet.elastic.manager.
                         ELASTIC_AUTO_PARALLEL_EXIT_CODE)

            original_cmd_args = os.getenv("PADDLE_ORIGINAL_CMD_ARGS")
            rank_mapping_args = " ".join(
                ["--rank_mapping_path", self._rank_mapping_path])
            if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
                coverage_args = ["-m", "coverage", "run", "--branch", "-p"]
            else:
                coverage_args = []
            new_cmd_args = "-m paddle.distributed.fleet.launch" + " " + rank_mapping_args + " " + original_cmd_args
            new_cmd = [sys.executable, "-u"] + coverage_args + shlex.split(
                new_cmd_args)
            new_process = subprocess.Popen(new_cmd)
            new_process.wait()
            assert new_process.returncode == 0, \
                "Launch failed with rank mapping"
            print("Successfully do the second launch for auto mapping!")
            sys.exit(0)
        else:
            # Parallelization after the mapping pass
            rank = paddle.distributed.get_rank()

            dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog = self._get_dist_program(
                self._dist_context, rank)

            # Traverse different rank programs and traverse each op of them,
            # instantiate communication by process_mapping.
            all_process_groups = get_all_process_groups()
            for process_group in all_process_groups:
                if rank not in process_group.ranks:
                    continue
                process_group.instantiate()

            # Copy distributed info to the default context
            set_default_distributed_context(self._dist_context)

            # The last step: remove all distributed attributes to be compatible
            # with inference.
            self._remove_distributed_attrs(dist_main_prog)

            return dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog
