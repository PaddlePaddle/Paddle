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

from .trainer_desc import MultiTrainer, DistMultiTrainer
from .device_worker import Hogwild, DownpourSGD

__all__ = ["TrainerFactory"]


class TrainerFactory(object):
    def __init__(self):
        pass

    def create_trainer(self, opt_info=None):
        trainer = None
        device_worker = None
        if opt_info == None:
            # default is MultiTrainer + Hogwild
            trainer = MultiTrainer()
            device_worker = Hogwild()
            trainer.set_device_worker(device_worker)
        else:
            trainer_class = opt_info["trainer"]
            device_worker_class = opt_info["device_worker"]
            trainer = globals()[trainer_class]()
            device_worker = globals()[device_worker_class]()
            device_worker.set_fleet_desc(opt_info["fleet_desc"])
            trainer.set_device_worker(device_worker)
            trainer.set_fleet_desc(opt_info["fleet_desc"])
        return trainer
