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

__all__ = ['TrainerDesc', 'MultiTrainer', 'DistMultiTrainer', 'PipelineTrainer']


# can be initialized from train_desc,
class TrainerDesc(object):
    def __init__(self):
        '''
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        '''
        from proto import trainer_desc_pb2
        self.proto_desc = trainer_desc_pb2.TrainerDesc()
        import multiprocessing as mp
        # set default thread num == cpu count
        self.proto_desc.thread_num = mp.cpu_count()
        self._fleet_desc = None
        self._device_worker = None
        self._program = None
        self._infer = False

    def _set_fetch_var_and_info(self, fetch_vars, fetch_info, print_period):
        for i, v in enumerate(fetch_vars):
            self.proto_desc.fetch_config.fetch_var_names.extend([v.name])
            self.proto_desc.fetch_config.fetch_var_str_format.extend(
                [fetch_info[i]])
        self.proto_desc.fetch_config.print_period = print_period

    def _set_debug(self, debug):
        self.proto_desc.debug = debug

    def _set_thread(self, thread_num):
        self.proto_desc.thread_num = thread_num

    def _set_device_worker(self, device_worker):
        self._device_worker = device_worker

    def _set_infer(self, infer):
        self._infer = infer

    def _set_fleet_desc(self, fleet_desc):
        self._fleet_desc = fleet_desc

    def _gen_trainer_desc(self):
        pass

    def _set_program(self, program):
        self._program = program

    def _set_use_cvm(self, use_cvm=False):
        self.proto_desc.use_cvm = use_cvm

    def _desc(self):
        from google.protobuf import text_format
        return self.proto_desc.SerializeToString()


class MultiTrainer(TrainerDesc):
    def __init__(self):
        super(MultiTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(MultiTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(MultiTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "MultiTrainer"
        self._device_worker._set_infer(self._infer)
        self._device_worker._gen_worker_desc(self.proto_desc)


class DistMultiTrainer(TrainerDesc):
    def __init__(self):
        super(DistMultiTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(DistMultiTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(DistMultiTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "DistMultiTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)


class PipelineTrainer(TrainerDesc):
    def __init__(self):
        super(PipelineTrainer, self).__init__()
        pass

    def _set_program(self, program):
        super(PipelineTrainer, self)._set_program(program)
        self._program = program

    def _gen_trainer_desc(self):
        super(PipelineTrainer, self)._gen_trainer_desc()
        self.proto_desc.class_name = "PipelineTrainer"
        if self._program == None:
            raise RuntimeError("None Program")
        self._device_worker._set_infer(self._infer)
        self._device_worker._set_program(self._program)
        self._device_worker._gen_worker_desc(self.proto_desc)
