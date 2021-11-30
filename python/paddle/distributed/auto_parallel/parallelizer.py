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
import pickle
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
from .auto_search import auto_search

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

    def _remove_distributed_attrs(self, main_program):
        suffix = core.kAutoParallelSuffix()
        # distributed attributes for variable have been removed
        # in previous process.
        for block in main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    if suffix in attr_name:
                        op._remove_attr(attr_name)

<<<<<<< HEAD
    def _get_dist_program(self, rank, dist_context=None):
        completed_main_program = None
        if dist_context is None:
=======
    def parallelize(self,
                    loss,
                    startup_program,
                    parameter_list=None,
                    no_grad_set=None):
        assert startup_program is not None
        main_program = loss.block.program

        if self._dist_strategy.auto_search:
            # auto search
            _logger.info("Start search dist attr.")
            self._dist_context, _ = auto_search(main_program, startup_program,
                                                loss, self._optimizer)
            completed_main_program = main_program
        else:
>>>>>>> close cost model
            # Annotation completion
            self._dist_context = DistributedContext()
            _logger.info("Start annotation dist attr.")
            completed_main_program = complete_annotation(self._main_program,
                                                         self._dist_context)
        else:
            completed_main_program = self._main_program
            self._dist_context = copy.deepcopy(dist_context)

        # Logical partition
        partitioner = Partitioner(self._dist_strategy, self._dist_context, rank)
        dist_main_prog, dist_startup_prog = partitioner.transpile_forward(
            completed_main_program, self._startup_program)
        dist_params_grads = partitioner.apply_backward(
            self._loss, completed_main_program, self._startup_program,
            dist_main_prog, dist_startup_prog)
        dist_optimize_ops = partitioner.apply_optimize(
            copy.deepcopy(self._optimizer), dist_params_grads, dist_main_prog,
            dist_startup_prog)

        set_grad_var_shape(dist_main_prog, self._dist_context)

        make_data_unshard(dist_main_prog, dist_startup_prog, self._dist_context)

        reshard(dist_main_prog, dist_startup_prog, rank, self._dist_context)

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

        if self._enable_auto_mapping and self._rank_mapping_path is None:
            # Do the mapping pass before parallelization
            assert self._cluster is not None, \
                "The cluster must not be none when using auto mapping."
            dist_programs = {}
            world_process_group = get_world_process_groups()
            dist_context = None
            # auto_search
            if self._dist_strategy.auto_search:
                _logger.info("Start search dist attr.")
                dist_context, _ = auto_search(self._main_program, self._startup_program, loss, self._optimizer)
            
            for rank in world_process_group.ranks:
                dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog = self._get_dist_program(
                    rank, dist_context)
                dist_programs[rank] = dist_main_prog

            # Do the mapping between the distributed program graph and the cluster graph
            rank_mapping_dict = mapping(dist_programs, self._cluster)
            rank_mapping = list(rank_mapping_dict.values())

            # Relaunch the training by using the rank mapping file
            cwd = pathlib.Path().resolve()
            rank_mapping_path = os.path.join(cwd,
                                             "auto_parallel_rank_mapping.json")
            with open(rank_mapping_path, "w") as rank_mapping_file:
                json.dump(rank_mapping, rank_mapping_file)

            # serialize the dist_context by planner
            if dist_context is not None:
                searched_dist_context_path = os.path.join(cwd, f"searched_dist_context_{time.time()}.pkl")
                with open(searched_dist_context_path, "wb") as dist_context_file:
                    pickle.dump(dist_context, dist_context_file)
                    os.environ['PADDLE_SEARCHED_DIST_CONTEXT_PATH'] = searched_dist_context_path

            original_cmd_args = os.getenv("PADDLE_ORIGINAL_CMD_ARGS")
            rank_mapping_args = " ".join(
                ["--rank_mapping_path", rank_mapping_path])
            new_cmd_args = "-u -m paddle.distributed.fleet.launch" + " " + rank_mapping_args + " " + original_cmd_args
            new_cmd = [sys.executable] + shlex.split(new_cmd_args)
            print(new_cmd)
            new_process = subprocess.Popen(new_cmd)
            new_process.wait()
            assert new_process.returncode == 0, \
                "Launch failed with rank mapping"
            print("Successfully do the second launch for auto mapping!")
            sys.exit(0)
        else:
            # Parallelization after the mapping pass
            rank = paddle.distributed.get_rank()
            dist_context = None
            searched_dist_context_path = os.getenv("PADDLE_SEARCHED_DIST_CONTEXT_PATH", None)
            if searched_dist_context_path is not None:
                with open(searched_dist_context_path, "rb") as dist_context_file:
                    dist_context = pickle.load(dist_context_file)
            else:
                if self._dist_strategy.auto_search:
                    _logger.info("Start search dist attr.")
                    dist_context, _ = auto_search(self._main_program, self._startup_program, loss, self._optimizer)
            dist_optimize_ops, dist_params_grads, dist_startup_prog, dist_main_prog = self._get_dist_program(rank, dist_context)

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
