#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.proto import trainer_desc_pb2
from google.protobuf import text_format

__all__ = ['TrainerDesc', 'MultiTrainer', 'DistMultiTrainer']


# can be initialized from train_desc, 
class TrainerDesc(object):
    def __init__(self):
        '''
        self.proto_desc = data_feed_pb2.DataFeedDesc()
        with open(proto_file, 'r') as f:
            text_format.Parse(f.read(), self.proto_desc)
        '''
        self.proto_desc = trainer_desc_pb2.TrainerDesc()

    def set_thread(self, thread_num):
        self.proto_desc.thread_num = thread_num

    def set_filelist(self, filelist):
        self.proto_desc.filelist.extend(filelist)

    def set_data_feed(self, datafeed):
        self.proto_desc.data_desc.CopyFrom(datafeed.proto_desc)

    def _desc(self):
        return text_format.MessageToString(self.proto_desc)


class MultiTrainer(TrainerDesc):
    def __init__(self, worker="Hogwild"):
        super(MultiTrainer, self).__init__()
        if worker == "Hogwild":
            self.proto_desc.device_worker_name = worker + "Worker"
            self.proto_desc.class_name = "MultiTrainer"
        else:
            raise ValueError('ValueError: DeviceWorker %s '
                             'is not supported in MultiTrainer' % worker)


class DistMultiTrainer(TrainerDesc):
    def __init__(self, worker='Downpour'):
        super(DistMultiTrainer, self).__init__()
        if worker == "Downpour":
            self.proto_desc.device_worker_name = worker + "Worker"
            self.proto_desc.class_name = "DistMultiTrainer"
        else:
            raise ValueError('ValueError: DeviceWorker %s '
                             'is not supported in DistMultiTrainer' % worker)
