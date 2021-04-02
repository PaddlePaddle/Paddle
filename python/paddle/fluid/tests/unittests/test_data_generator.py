#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import unittest
import paddle.distributed.fleet as fleet
import os
import sys
import platform
from paddle.dataset.common import download, DATA_HOME


class MyMultiSlotDataGenerator(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", [1, 2, 3, 4]), ("label", [0])

        return data_iter


class MyMultiSlotStringDataGenerator(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", ["1", "2", "3", "4"]), ("label", ["0"])

        return data_iter


class MyMultiSlotDataGenerator_error(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield "words"

        return data_iter


class MyMultiSlotDataGenerator_error_2(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield "words"

        return data_iter


class MyMultiSlotDataGenerator_error_3(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield (1, ["1", "2", "3", "4"]), (2, ["0"])

        return data_iter


class MyMultiSlotDataGenerator_error_4(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", "1"), ("label", "0")

        return data_iter


class MyMultiSlotDataGenerator_error_5(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", []), ("label", [])

        return data_iter


class MyMultiSlotStringDataGenerator_zip(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                feature_name = ["words", "label"]
                data = [["1", "2", "3", "4"], ["0"]]
                yield zip(feature_name, data)

        return data_iter


class MyMultiSlotDataGenerator_zip(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                feature_name = ["words", "label"]
                data = [[1, 2, 3, 4], [0]]
                yield zip(feature_name, data)

        return data_iter


class TestMultiSlotDataGenerator(unittest.TestCase):
    def test_MultiSlotDataGenerator_basic(self):
        my_ms_dg = MyMultiSlotDataGenerator()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class TestMultiSlotStringDataGenerator(unittest.TestCase):
    def test_MyMultiSlotStringDataGenerator_basic(self):
        my_ms_dg = MyMultiSlotStringDataGenerator()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_2(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_2()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_3(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_3()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_4(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_4()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_5(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_5()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotStringDataGeneratorZip(unittest.TestCase):
    def test_MultiSlotStringDataGenerator_zip(self):
        my_ms_dg = MyMultiSlotStringDataGenerator_zip()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class TestMultiSlotDataGeneratorZip(unittest.TestCase):
    def test_MultiSlotDataGenerator_zip(self):
        my_ms_dg = MyMultiSlotDataGenerator_zip()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class DemoTreeIndexDataset(fleet.MultiSlotDataGenerator):
    def init(self):
        self.item_nums = 69

    def line_process(self, line):
        result = [[0]] * (self.item_nums + 2)
        features = line.strip().split("\t")
        item_id = int(features[1])
        for item in features[2:]:
            slot, feasign = item.split(":")
            slot_id = int(slot.split("_")[1])
            result[slot_id - 1] = [int(feasign)]
        result[-2] = [item_id]
        result[-1] = [1]
        return result

    def generate_sample(self, line):
        "Dataset Generator"

        def reader():
            demo_line = "1000186_8\t949301\tslot_55:1571034\tslot_56:4780300\tslot_57:1744498\tslot_58:3597221\tslot_59:2906708\tslot_60:901439\tslot_61:2793602\tslot_62:1652215\tslot_63:4682748\tslot_64:823068\tslot_65:3395014\tslot_66:369520\tslot_67:3395014\tslot_68:4498543\tslot_69:4048294"
            output_list = self.line_process(demo_line)
            feature_name = []
            for i in range(self.item_nums):
                feature_name.append("item_" + str(i + 1))
            feature_name.append("unit_id")
            feature_name.append("label")
            yield list(zip(feature_name, output_list))

        return reader


class TestTreeIndexDataGenerator(unittest.TestCase):
    def test_TreeIndexDataGenerator(self):
        dataset = DemoTreeIndexDataset()
        dataset.init()

        path = download(
            "https://paddlerec.bj.bcebos.com/tree-based/data/demo_tree.pb",
            "tree_index_unittest", "cadec20089f5a8a44d320e117d9f9f1a")

        tree = fleet.data_generator.TreeIndex("demo", path)
        dataset.set_tree_layerwise_sampler(
            "demo", [1] * 14, range(69), 69, 70, with_hierarchy=True)
        dataset.set_batch(1)
        dataset.run_from_memory()


if __name__ == '__main__':
    unittest.main()
