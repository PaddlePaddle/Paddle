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
"""
TestCases for Dataset,
including create, config, run, etc.
"""

import os
import tempfile
import unittest

import paddle
from paddle import base
from paddle.base import core


class TestDataset(unittest.TestCase):
    """TestCases for Dataset."""

    def setUp(self):
        self.use_data_loader = False
        self.epoch_num = 10
        self.drop_last = False

    def test_dataset_create(self):
        """Testcase for dataset create."""
        try:
            dataset = paddle.distributed.InMemoryDataset()
        except:
            self.assertTrue(False)

        try:
            dataset = paddle.distributed.QueueDataset()
        except:
            self.assertTrue(False)

        try:
            dataset = paddle.distributed.fleet.dataset.FileInstantDataset()
        except:
            self.assertTrue(False)

        try:
            dataset = paddle.distributed.fleet.dataset.MyOwnDataset()
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_config(self):
        """
        Testcase for python config.
        """
        dataset = base.InMemoryDataset()
        dataset.set_parse_ins_id(True)
        dataset.set_parse_content(True)
        dataset._set_trainer_num(1)
        self.assertTrue(dataset.parse_ins_id)
        self.assertTrue(dataset.parse_content)
        self.assertEqual(dataset.trainer_num, 1)

    def test_shuffle_by_uid(self):
        """
        Testcase for shuffle_by_uid.
        """
        dataset = paddle.distributed.InMemoryDataset()
        dataset._set_uid_slot('6048')
        dataset._set_shuffle_by_uid(True)

    def test_run_with_dump(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            dump_a_path = os.path.join(
                temp_dir.name, 'test_run_with_dump_a.txt'
            )
            dump_b_path = os.path.join(
                temp_dir.name, 'test_run_with_dump_b.txt'
            )

            with open(dump_a_path, "w") as f:
                data = "1 a 1 a 1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 b 1 b 1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 c 1 c 1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(dump_b_path, "w") as f:
                data = "1 d 1 d 1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 e 1 e 1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 f 1 f 1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 g 1 g 1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.update_settings(pipe_command="cat1")
            dataset._init_distributed_settings(
                parse_ins_id=True,
                parse_content=True,
                fea_eval=True,
                candidate_size=10000,
            )
            dataset.set_filelist([dump_a_path, dump_b_path])
            dataset.load_into_memory()
            dataset.local_shuffle()

            paddle.enable_static()

            exe = paddle.static.Executor(paddle.CPUPlace())
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            exe.run(startup_program)
            for i in range(2):
                try:
                    exe.train_from_dataset(main_program, dataset)
                except ImportError as e:
                    pass
                except Exception as e:
                    self.assertTrue(False)

            temp_dir.cleanup()

    def test_dataset_config(self):
        """Testcase for dataset configuration."""
        dataset = base.core.Dataset("MultiSlotDataset")
        dataset.set_thread_num(12)
        dataset.set_filelist(["a.txt", "b.txt", "c.txt"])
        dataset.set_trainer_num(4)
        dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")
        dataset.set_download_cmd("./read_from_afs my_fs_name my_fs_ugi")
        dataset.set_enable_pv_merge(False)

        thread_num = dataset.get_thread_num()
        self.assertEqual(thread_num, 12)

        filelist = dataset.get_filelist()
        self.assertEqual(len(filelist), 3)
        self.assertEqual(filelist[0], "a.txt")
        self.assertEqual(filelist[1], "b.txt")
        self.assertEqual(filelist[2], "c.txt")

        trainer_num = dataset.get_trainer_num()
        self.assertEqual(trainer_num, 4)

        name, ugi = dataset.get_hdfs_config()
        self.assertEqual(name, "my_fs_name")
        self.assertEqual(ugi, "my_fs_ugi")

        download_cmd = dataset.get_download_cmd()
        self.assertEqual(download_cmd, "./read_from_afs my_fs_name my_fs_ugi")

    def test_set_download_cmd(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "afs:test_in_memory_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "afs:test_in_memory_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                download_cmd="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            paddle.enable_static()

            exe = paddle.static.Executor(paddle.CPUPlace())
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            exe = base.Executor(base.CPUPlace())
            exe.run(startup_program)
            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(main_program, feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(main_program, dataset)
                    except Exception as e:
                        self.assertTrue(False)

            temp_dir.cleanup()

    def test_in_memory_dataset_run(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset._init_distributed_settings(fea_eval=True, candidate_size=1)
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            dataset.slots_shuffle(["slot1"])
            dataset.local_shuffle()
            dataset._set_generate_unique_feasigns(True, 15)
            dataset._generate_local_tables_unlock(0, 11, 1, 25, 15)
            exe = base.Executor(base.CPUPlace())
            exe.run(base.default_startup_program())
            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(base.default_main_program(), feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(
                            base.default_main_program(), dataset
                        )
                    except Exception as e:
                        self.assertTrue(False)

            temp_dir.cleanup()

    def test_in_memory_dataset_gpugraph_mode(self):
        """
        Testcase for InMemoryDataset in gpugraph mode.
        """
        dataset = base.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("SlotRecordInMemoryDataFeed")
        graph_config = {
            "walk_len": 24,
            "walk_degree": 10,
            "once_sample_startid_len": 80000,
            "sample_times_one_chunk": 5,
            "window": 3,
            "debug_mode": 0,
            "batch_size": 800,
            "meta_path": "cuid2clk-clk2cuid;cuid2conv-conv2cuid;clk2cuid-cuid2clk;clk2cuid-cuid2conv",
            "gpu_graph_training": 1,
        }
        dataset.set_graph_config(graph_config)
        dataset.set_pass_id(0)
        dataset.get_pass_id()
        dataset.get_epoch_finish()

    def test_in_memory_dataset_masterpatch(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_masterpatch_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_masterpatch_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 id1 1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 id1 1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 id2 1 1 1 1 1 0 1 0\n"
                data += "1 id3 1 0 1 0 1 1 1 1\n"
                data += "1 id3 1 1 1 1 1 0 1 0\n"
                data += "1 id4 1 0 1 0 1 1 1 1\n"
                data += "1 id4 1 0 1 0 1 1 1 1\n"
                data += "1 id5 1 1 1 1 1 0 1 0\n"
                data += "1 id5 1 1 1 1 1 0 1 0\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 id6 1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 id6 1 1 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 id6 1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 id6 1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            train_program = base.Program()
            startup_program = base.Program()
            with base.program_guard(train_program, startup_program):
                for slot in slots[:2]:
                    var = paddle.static.data(
                        name=slot, shape=[-1, 1], dtype="int64"
                    )
                    slots_vars.append(var)
                for slot in slots[2:]:
                    var = paddle.static.data(
                        name=slot, shape=[-1, 1], dtype="float32"
                    )
                    slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()

            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset._init_distributed_settings(parse_ins_id=True)
            dataset.set_filelist(
                [
                    "test_in_memory_dataset_masterpatch_a.txt",
                    "test_in_memory_dataset_masterpatch_b.txt",
                ]
            )
            dataset.load_into_memory()
            dataset.local_shuffle()

            exe = base.Executor(base.CPUPlace())
            exe.run(startup_program)

            for i in range(2):
                try:
                    exe.train_from_dataset(train_program, dataset)
                except ImportError as e:
                    pass
                except Exception as e:
                    self.assertTrue(False)

            # dataset._set_merge_by_lineid(2)
            dataset.update_settings(merge_size=2)
            dataset.dataset.merge_by_lineid()
            temp_dir.cleanup()

    def test_in_memory_dataset_masterpatch1(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_masterpatch1_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_masterpatch1_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 id1 1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 id1 1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 id2 1 1 1 1 1 0 1 0\n"
                data += "1 id3 1 0 1 0 1 1 1 1\n"
                data += "1 id3 1 1 1 1 1 0 1 0\n"
                data += "1 id4 1 0 1 0 1 1 1 1\n"
                data += "1 id4 1 0 1 0 1 1 1 1\n"
                data += "1 id5 1 1 1 1 1 0 1 0\n"
                data += "1 id5 1 1 1 1 1 0 1 0\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 id6 1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 id6 1 1 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 id6 1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 id6 1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots_vars = []
            train_program = base.Program()
            startup_program = base.Program()
            with base.program_guard(train_program, startup_program):
                var1 = paddle.static.data(
                    name="slot1", shape=[-1, 1], dtype="int64"
                )
                var2 = paddle.static.data(
                    name="slot2", shape=[-1, 1], dtype="int64"
                )
                var3 = paddle.static.data(
                    name="slot3", shape=[-1, 1], dtype="float32"
                )
                var4 = paddle.static.data(
                    name="slot4", shape=[-1, 1], dtype="float32"
                )
                slots_vars = [var1, var2, var3, var4]

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset._init_distributed_settings(parse_ins_id=True)
            dataset.set_filelist(
                [
                    "test_in_memory_dataset_masterpatch1_a.txt",
                    "test_in_memory_dataset_masterpatch1_b.txt",
                ]
            )
            dataset.load_into_memory()
            dataset.local_shuffle()

            exe = base.Executor(base.CPUPlace())
            exe.run(startup_program)

            for i in range(2):
                try:
                    exe.train_from_dataset(train_program, dataset)
                except ImportError as e:
                    pass
                except Exception as e:
                    self.assertTrue(False)

            dataset._set_merge_by_lineid(2)
            dataset.dataset.merge_by_lineid()

            temp_dir.cleanup()

    def test_in_memory_dataset_run_2(self):
        """
        Testcase for InMemoryDataset from create to run.
        Use CUDAPlace
        Use float type id
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1_f", "slot2_f", "slot3_f", "slot4_f"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="float32"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            dataset.local_shuffle()

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(base.default_startup_program())

            for i in range(2):
                try:
                    exe.train_from_dataset(base.default_main_program(), dataset)
                    # exe.train_from_dataset(
                    #     base.default_main_program(), dataset, thread=1
                    # )
                    exe.train_from_dataset(
                        base.default_main_program(), dataset, thread=2
                    )
                    # exe.train_from_dataset(
                    #     base.default_main_program(), dataset, thread=2
                    # )
                    # exe.train_from_dataset(
                    #     base.default_main_program(), dataset, thread=3
                    # )
                    # exe.train_from_dataset(
                    #     base.default_main_program(), dataset, thread=4
                    # )
                except ImportError as e:
                    pass
                except Exception as e:
                    self.assertTrue(False)

            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(base.default_main_program(), feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(
                            base.default_main_program(), dataset
                        )
                    except Exception as e:
                        self.assertTrue(False)

            dataset._set_merge_by_lineid(2)
            dataset._set_parse_ins_id(False)
            dataset._set_fleet_send_sleep_seconds(2)
            dataset.preload_into_memory()
            dataset.wait_preload_done()
            dataset.preload_into_memory(1)
            dataset.wait_preload_done()
            dataset.dataset.merge_by_lineid()
            dataset._set_merge_by_lineid(30)
            dataset._set_parse_ins_id(False)
            dataset.load_into_memory()
            dataset.dataset.merge_by_lineid()
            dataset.update_settings(
                batch_size=1,
                thread_num=2,
                input_type=1,
                pipe_command="cat",
                use_var=[],
                fs_name="",
                fs_ugi="",
                download_cmd="cat",
                merge_size=-1,
                parse_ins_id=False,
                parse_content=False,
                fleet_send_batch_size=2,
                fleet_send_sleep_seconds=2,
                fea_eval=True,
            )
            fleet_ptr = base.core.Fleet()
            fleet_ptr.set_client2client_config(1, 1, 1)
            fleet_ptr.get_cache_threshold(0)

            temp_dir.cleanup()

    def test_queue_dataset_run(self):
        """
        Testcase for QueueDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.QueueDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])

            exe = base.Executor(base.CPUPlace())
            exe.run(base.default_startup_program())
            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(base.default_main_program(), feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(
                            base.default_main_program(), dataset
                        )
                    except Exception as e:
                        self.assertTrue(False)

            dataset2 = paddle.distributed.QueueDataset()
            dataset2.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([])
            # try:
            #    exe.train_from_dataset(base.default_main_program(), dataset2)
            # except ImportError as e:
            #    print("warning: we skip trainer_desc_pb2 import problem in windows")
            # except Exception as e:
            #    self.assertTrue(False)

            temp_dir.cleanup()

    def test_queue_dataset_run_2(self):
        """
        Testcase for QueueDataset from create to run.
        Use CUDAPlace
        Use float type id
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1_f", "slot2_f", "slot3_f", "slot4_f"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="float32"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.QueueDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(base.default_startup_program())
            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(base.default_main_program(), feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(
                            base.default_main_program(), dataset
                        )
                    except Exception as e:
                        self.assertTrue(False)

            temp_dir.cleanup()

    def test_queue_dataset_run_3(self):
        """
        Testcase for QueueDataset from create to run.
        Use CUDAPlace
        Use float type id
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_queue_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
                data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
                data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
                data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "2 1 2 2 5 4 2 2 7 2 1 3\n"
                data += "2 6 2 2 1 4 2 2 4 2 2 3\n"
                data += "2 5 2 2 9 9 2 2 7 2 1 3\n"
                data += "2 7 2 2 1 9 2 3 7 2 5 3\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[None, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=1,
                thread_num=2,
                input_type=1,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(base.default_startup_program())
            if self.use_data_loader:
                data_loader = base.io.DataLoader.from_dataset(
                    dataset, base.cpu_places(), self.drop_last
                )
                for i in range(self.epoch_num):
                    for data in data_loader():
                        exe.run(base.default_main_program(), feed=data)
            else:
                for i in range(self.epoch_num):
                    try:
                        exe.train_from_dataset(
                            base.default_main_program(), dataset
                        )
                    except Exception as e:
                        self.assertTrue(False)

            temp_dir.cleanup()

    def test_run_with_inmemory_dataset_train_debug_mode(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            dump_a_path = os.path.join(
                temp_dir.name, 'test_run_with_dump_a.txt'
            )
            dump_b_path = os.path.join(
                temp_dir.name, 'test_run_with_dump_b.txt'
            )

            with open(dump_a_path, "w") as f:
                data = "1 a 1 a 1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 b 1 b 1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 c 1 c 1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(dump_b_path, "w") as f:
                data = "1 d 1 d 1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 e 1 e 1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 f 1 f 1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 g 1 g 1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                data_feed_type="SlotRecordInMemoryDataFeed",
                use_var=slots_vars,
            )
            dataset._init_distributed_settings(
                parse_ins_id=True,
                parse_content=True,
                fea_eval=True,
                candidate_size=10000,
            )
            dataset.set_filelist([dump_a_path, dump_b_path])
            dataset.load_into_memory()

            paddle.enable_static()

            exe = paddle.static.Executor(paddle.CPUPlace())
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            exe.run(startup_program)
            for i in range(2):
                try:
                    exe.train_from_dataset(main_program, dataset, debug=True)
                except ImportError as e:
                    pass
                except Exception as e:
                    self.assertTrue(False)

            temp_dir.cleanup()

    def test_cuda_in_memory_dataset_run(self):
        """
        Testcase for cuda inmemory dataset hogwild_worker train to run(barrier).
        """
        with paddle.pir_utils.OldIrGuard():
            temp_dir = tempfile.TemporaryDirectory()
            filename1 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_a.txt"
            )
            filename2 = os.path.join(
                temp_dir.name, "test_in_memory_dataset_run_b.txt"
            )

            with open(filename1, "w") as f:
                data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
                data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
                data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
                f.write(data)
            with open(filename2, "w") as f:
                data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
                data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
                data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
                data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
                f.write(data)

            slots = ["slot1", "slot2", "slot3", "slot4"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="int64"
                )
                slots_vars.append(var)

            dataset = base.DatasetFactory().create_dataset("InMemoryDataset")
            dataset.set_feed_type("SlotRecordInMemoryDataFeed")
            dataset.set_batch_size(1)
            dataset.set_pipe_command("cat")
            dataset.set_use_var(slots_vars)
            dataset.set_filelist([filename1, filename2])

            dataset.set_pass_id(2)
            pass_id = dataset.get_pass_id()

            dataset.set_thread(2)
            dataset.load_into_memory()

            dataset.get_memory_data_size()

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(base.default_startup_program())
            for i in range(self.epoch_num):
                try:
                    exe.train_from_dataset(base.default_main_program(), dataset)
                except Exception as e:
                    self.assertTrue(False)
            temp_dir.cleanup()


class TestDatasetWithDataLoader(TestDataset):
    """
    Test Dataset With Data Loader class. TestCases.
    """

    def setUp(self):
        """
        Test Dataset With Data Loader, setUp.
        """
        self.use_data_loader = True
        self.epoch_num = 10
        self.drop_last = False


class TestDataset2(unittest.TestCase):
    """TestCases for Dataset."""

    def setUp(self):
        """TestCases for Dataset."""
        self.use_data_loader = False
        self.epoch_num = 10
        self.drop_last = False

    def test_dataset_fleet(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        temp_dir = tempfile.TemporaryDirectory()
        filename1 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run_a.txt"
        )
        filename2 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run_b.txt"
        )

        self.skipTest("parameter server will add pslib UT later")

        with open(filename1, "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open(filename2, "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        from paddle.incubate.distributed.fleet.parameter_server.distribute_transpiler import (
            fleet,
        )

        with base.program_guard(train_program, startup_program):
            slots = ["slot1_ff", "slot2_ff", "slot3_ff", "slot4_ff"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="float32"
                )
                slots_vars.append(var)
            fake_cost = paddle.subtract(slots_vars[0], slots_vars[-1])
            fake_cost = paddle.mean(fake_cost)
        with base.scope_guard(scope):
            place = base.CPUPlace()
            exe = base.Executor(place)
            try:
                fleet.init()
            except ImportError as e:
                print("warning: no mpi4py")
            adam = paddle.optimizer.Adam(learning_rate=0.000005)
            try:
                adam = fleet.distributed_optimizer(adam)
                adam.minimize([fake_cost], [scope])
            except AttributeError as e:
                print("warning: no mpi")
            except ImportError as e:
                print("warning: no mpi4py")
            exe.run(startup_program)
            dataset = paddle.distributed.InMemoryDataset()

            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            fleet._opt_info = None
            fleet._fleet_ptr = None

        temp_dir.cleanup()

    def test_dataset_fleet2(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        temp_dir = tempfile.TemporaryDirectory()
        filename1 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run2_a.txt"
        )
        filename2 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run2_b.txt"
        )

        with open(filename1, "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open(filename2, "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        from paddle.incubate.distributed.fleet.parameter_server.pslib import (
            fleet,
        )

        with base.program_guard(train_program, startup_program):
            slots = ["slot1_ff", "slot2_ff", "slot3_ff", "slot4_ff"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="float32"
                )
                slots_vars.append(var)
            fake_cost = paddle.subtract(slots_vars[0], slots_vars[-1])
            fake_cost = paddle.mean(fake_cost)
        with base.scope_guard(scope):
            place = base.CPUPlace()
            exe = base.Executor(place)
            try:
                fleet.init()
            except ImportError as e:
                print("warning: no mpi4py")
            adam = paddle.optimizer.Adam(learning_rate=0.000005)
            try:
                adam = fleet.distributed_optimizer(
                    adam,
                    strategy={
                        "fs_uri": "fs_uri_xxx",
                        "fs_user": "fs_user_xxx",
                        "fs_passwd": "fs_passwd_xxx",
                        "fs_hadoop_bin": "fs_hadoop_bin_xxx",
                    },
                )
                adam.minimize([fake_cost], [scope])
            except AttributeError as e:
                print("warning: no mpi")
            except ImportError as e:
                print("warning: no mpi4py")
            exe.run(startup_program)
            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            try:
                dataset.global_shuffle(fleet)
            except:
                print("warning: catch expected error")
            fleet._opt_info = None
            fleet._fleet_ptr = None
            dataset = paddle.distributed.InMemoryDataset()
            dataset.init(fs_name="", fs_ugi="")
            d = paddle.distributed.fleet.DatasetBase()
            try:
                dataset._set_feed_type("MultiSlotInMemoryDataFeed")
            except:
                print("warning: catch expected error")
            dataset.thread_num = 0
            try:
                dataset._prepare_to_run()
            except:
                print("warning: catch expected error")
            try:
                dataset.preprocess_instance()
            except:
                print("warning: catch expected error")
            try:
                dataset.set_current_phase(1)
            except:
                print("warning: catch expected error")
            try:
                dataset.postprocess_instance()
            except:
                print("warning: catch expected error")
            dataset._set_fleet_send_batch_size(1024)
            try:
                dataset.global_shuffle()
            except:
                print("warning: catch expected error")
            # dataset.get_pv_data_size()
            dataset.get_memory_data_size()
            dataset.get_shuffle_data_size()
            dataset = paddle.distributed.QueueDataset()
            try:
                dataset.local_shuffle()
            except:
                print("warning: catch expected error")
            try:
                dataset.global_shuffle()
            except:
                print("warning: catch expected error")
            dataset = paddle.distributed.fleet.FileInstantDataset()
            try:
                dataset.local_shuffle()
            except:
                print("warning: catch expected error")
            try:
                dataset.global_shuffle()
            except:
                print("warning: catch expected error")

        temp_dir.cleanup()

    def test_bosps_dataset_fleet2(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        temp_dir = tempfile.TemporaryDirectory()
        filename1 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run2_a.txt"
        )
        filename2 = os.path.join(
            temp_dir.name, "test_in_memory_dataset2_run2_b.txt"
        )

        with open(filename1, "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open(filename2, "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        from paddle.incubate.distributed.fleet.parameter_server.pslib import (
            fleet,
        )

        with base.program_guard(train_program, startup_program):
            slots = ["slot1_ff", "slot2_ff", "slot3_ff", "slot4_ff"]
            slots_vars = []
            for slot in slots:
                var = paddle.static.data(
                    name=slot, shape=[-1, 1], dtype="float32"
                )
                slots_vars.append(var)
            fake_cost = paddle.subtract(slots_vars[0], slots_vars[-1])
            fake_cost = paddle.mean(fake_cost)
        with base.scope_guard(scope):
            place = base.CPUPlace()
            exe = base.Executor(place)
            try:
                fleet.init()
            except ImportError as e:
                print("warning: no mpi4py")
            adam = paddle.optimizer.Adam(learning_rate=0.000005)
            try:
                adam = fleet.distributed_optimizer(
                    adam,
                    strategy={
                        "fs_uri": "fs_uri_xxx",
                        "fs_user": "fs_user_xxx",
                        "fs_passwd": "fs_passwd_xxx",
                        "fs_hadoop_bin": "fs_hadoop_bin_xxx",
                    },
                )
                adam.minimize([fake_cost], [scope])
            except AttributeError as e:
                print("warning: no mpi")
            except ImportError as e:
                print("warning: no mpi4py")
            exe.run(startup_program)
            dataset = paddle.distributed.fleet.BoxPSDataset()
            dataset.init(
                batch_size=32,
                thread_num=2,
                pipe_command="cat",
                use_var=slots_vars,
            )
            dataset.set_filelist([filename1, filename2])
            dataset.load_into_memory()
            try:
                dataset.global_shuffle(fleet)
            except:
                print("warning: catch expected error")
            fleet._opt_info = None
            fleet._fleet_ptr = None
            dataset = paddle.distributed.fleet.BoxPSDataset()
            dataset.init(
                rank_offset="",
                pv_batch_size=1,
                fs_name="",
                fs_ugi="",
                data_feed_type="MultiSlotInMemoryDataFeed",
                parse_logkey=True,
                merge_by_sid=True,
                enable_pv_merge=True,
            )
            d = paddle.distributed.fleet.DatasetBase()
            try:
                dataset._set_feed_type("MultiSlotInMemoryDataFeed")
            except:
                print("warning: catch expected error")
            dataset.thread_num = 0
            try:
                dataset._prepare_to_run()
            except:
                print("warning: catch expected error")
            dataset._set_parse_logkey(True)
            dataset._set_merge_by_sid(True)
            dataset._set_enable_pv_merge(True)
            try:
                dataset.preprocess_instance()
            except:
                print("warning: catch expected error")
            try:
                dataset.set_current_phase(1)
            except:
                print("warning: catch expected error")
            try:
                dataset.postprocess_instance()
            except:
                print("warning: catch expected error")
            dataset._set_fleet_send_batch_size(1024)
            try:
                dataset.global_shuffle()
            except:
                print("warning: catch expected error")
            # dataset.get_pv_data_size()
            dataset.get_memory_data_size()
            dataset.get_shuffle_data_size()
        temp_dir.cleanup()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
