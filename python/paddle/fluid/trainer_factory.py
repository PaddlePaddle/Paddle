#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Defination of TrainerFactory."""

import threading
import time

import numpy as np

from .trainer_desc import MultiTrainer, DistMultiTrainer, PipelineTrainer
from .device_worker import Hogwild, DownpourSGD, Section

__all__ = ["TrainerFactory", "FetchHandler", "FetchHandlerMonitor"]


class TrainerFactory(object):
    """
    Create trainer and device worker.
    If opt_info is not None, it will get configs from opt_info,
    otherwise create MultiTrainer and Hogwild.
    """

    def __init__(self):
        pass

    def _create_trainer(self, opt_info=None):
        trainer = None
        device_worker = None
        if opt_info == None:
            # default is MultiTrainer + Hogwild
            trainer = MultiTrainer()
            device_worker = Hogwild()
            trainer._set_device_worker(device_worker)
        else:
            trainer_class = opt_info["trainer"]
            device_worker_class = opt_info["device_worker"]
            trainer = globals()[trainer_class]()
            device_worker = globals()[device_worker_class]()
            if "fleet_desc" in opt_info:
                device_worker._set_fleet_desc(opt_info["fleet_desc"])
                trainer._set_fleet_desc(opt_info["fleet_desc"])
                if opt_info.get("use_cvm") is not None:
                    trainer._set_use_cvm(opt_info["use_cvm"])
                if opt_info.get("no_cvm") is not None:
                    trainer._set_no_cvm(opt_info["no_cvm"])
                if opt_info.get("scale_datanorm") is not None:
                    trainer._set_scale_datanorm(opt_info["scale_datanorm"])
                if opt_info.get("dump_slot") is not None:
                    trainer._set_dump_slot(opt_info["dump_slot"])
                if opt_info.get("mpi_rank") is not None:
                    trainer._set_mpi_rank(opt_info["mpi_rank"])
                if opt_info.get("mpi_size") is not None:
                    trainer._set_mpi_size(opt_info["mpi_size"])
                if opt_info.get("dump_fields") is not None:
                    trainer._set_dump_fields(opt_info["dump_fields"])
                if opt_info.get("dump_fields_path") is not None:
                    trainer._set_dump_fields_path(opt_info["dump_fields_path"])
                if opt_info.get("dump_file_num") is not None:
                    trainer._set_dump_file_num(opt_info["dump_file_num"])
                if opt_info.get("dump_converter") is not None:
                    trainer._set_dump_converter(opt_info["dump_converter"])
                if opt_info.get("adjust_ins_weight") is not None:
                    trainer._set_adjust_ins_weight(opt_info[
                        "adjust_ins_weight"])
                if opt_info.get("copy_table") is not None:
                    trainer._set_copy_table_config(opt_info["copy_table"])
                if opt_info.get("check_nan_var_names") is not None:
                    trainer._set_check_nan_var_names(opt_info[
                        "check_nan_var_names"])
                if opt_info.get("dump_param") is not None:
                    trainer._set_dump_param(opt_info["dump_param"])
            trainer._set_device_worker(device_worker)
        return trainer


class FetchHandlerMonitor(object):
    """
    Defination of FetchHandlerMonitor class,
    it's for fetch handler.
    """

    def __init__(self, scope, handler):
        self.fetch_instance = handler
        self.fetch_thread = threading.Thread(
            target=self.handler_decorator,
            args=(scope, self.fetch_instance.handler))
        self.running = False

    def start(self):
        """
        start monitor,
        it will start a monitor thread.
        """
        self.running = True
        self.fetch_thread.setDaemon(True)
        self.fetch_thread.start()

    def handler_decorator(self, fetch_scope, fetch_handler):
        """
        decorator of handler,
        Args:
            fetch_scope(Scope): fetch scope
            fetch_handler(Handler): fetch handler
        """
        fetch_target_names = self.fetch_instance.fetch_target_names
        period_secs = self.fetch_instance.period_secs

        elapsed_secs = 0
        while True:
            while self.running and elapsed_secs >= period_secs:
                elapsed_secs = 0

                fetch_vars = [
                    fetch_scope.find_var(varname)
                    for varname in fetch_target_names
                ]

                if None in fetch_vars:
                    continue

                fetch_tensors = [var.get_tensor() for var in fetch_vars]

                if self.fetch_instance.return_np:
                    fetch_nps = []

                    for tensor in fetch_tensors:
                        lod = tensor.lod()

                        if len(lod) > 0:
                            raise RuntimeError(
                                "Some of your fetched tensors hold LoD information. \
                        They can not be completely cast to Python ndarray. We can not \
                        return LoDTensor itself directly, please choose another targets"
                            )

                        if tensor._is_initialized():
                            fetch_nps.append(np.array(tensor))
                        else:
                            fetch_nps.append(None)

                    fetch_handler(fetch_nps)
                else:
                    fetch_handler(fetch_tensors)
            else:
                time.sleep(1)
                elapsed_secs += 1

    def stop(self):
        self.running = False
