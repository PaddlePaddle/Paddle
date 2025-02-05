# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed import fleet

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


class Reader(fleet.MultiSlotDataGenerator):
    def init(self):
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding
        logger.info("pipe init success")

    def line_process(self, line):
        line = line.strip().split(" ")
        output = [(i, []) for i in self.slots]
        for i in line:
            slot_feasign = i.split(":")
            slot = slot_feasign[0]
            if slot not in self.slots:
                continue
            if slot in self.sparse_slots:
                feasign = int(slot_feasign[1])
            else:
                feasign = float(slot_feasign[1])
            output[self.slot2index[slot]][1].append(feasign)
            self.visit[slot] = True
        for i in self.visit:
            slot = i
            if not self.visit[slot]:
                if i in self.dense_slots:
                    output[self.slot2index[i]][1].extend(
                        [self.padding]
                        * self.dense_slots_shape[self.slot2index[i]]
                    )
                else:
                    output[self.slot2index[i]][1].extend([self.padding])
            else:
                self.visit[slot] = False

        return output
        # return [label] + sparse_feature + [dense_feature]

    def generate_sample(self, line):
        r"Dataset Generator"

        def reader():
            output_dict = self.line_process(line)
            # {key, value} dict format: {'labels': [1], 'sparse_slot1': [2, 3], 'sparse_slot2': [4, 5, 6, 8], 'dense_slot': [1,2,3,4]}
            # dict must match static_model.create_feed()
            yield output_dict

        return reader


if __name__ == "__main__":
    r = Reader()
    r.init()
    r.run_from_stdin()
