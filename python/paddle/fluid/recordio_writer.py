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

import core


class RecordIOWriter(object):
    def __init__(self,
                 filename,
                 compressor=core.RecordIOWriter.Compressor.Snappy,
                 max_num_records=1000):
        self.filename = filename
        self.compressor = compressor
        self.max_num_records = max_num_records
        self.writer = None

    def __enter__(self):
        self.writer = core.RecordIOWriter(self.filename, self.compressor,
                                          self.max_num_records)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        else:
            self.writer.close()

    def append_tensor(self, tensor):
        self.writer.append_tensor(tensor)

    def complete_append_tensor(self):
        self.writer.complete_append_tensor()


def convert_reader_to_recordio_file(
        filename,
        reader_creator,
        feeder,
        compressor=core.RecordIOWriter.Compressor.Snappy,
        max_num_records=1000,
        feed_order=None):
    writer = RecordIOWriter(filename, compressor, max_num_records)
    with writer:
        for batch in reader_creator():
            res = feeder.feed(batch)
            if feed_order is None:
                for each in res:
                    writer.append_tensor(res[each])
            else:
                for each in feed_order:
                    writer.append_tensor(res[each])
            writer.complete_append_tensor()
