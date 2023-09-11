# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle
from paddle import base
from paddle.base.incubate.checkpoint.auto_checkpoint import ExeTrainStatus
from paddle.base.incubate.checkpoint.checkpoint_saver import CheckpointSaver
from paddle.distributed.fleet.utils.fs import HDFSClient, LocalFS
from paddle.incubate.distributed.fleet import role_maker
from paddle.incubate.distributed.fleet.collective import fleet


class FleetTest(unittest.TestCase):
    def _test_checkpoint(self, fs, dir_path):
        file_name = "persistables"

        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        image = paddle.static.data(
            name='img', shape=[None, 28, 28], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        feeder = base.DataFeeder(
            feed_list=[image, label], place=base.CPUPlace()
        )
        predict = paddle.static.nn.fc(x=image, size=10, activation='softmax')
        loss = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        optimizer = paddle.optimizer.Adam(learning_rate=0.001)

        dist_optimizer = fleet.distributed_optimizer(optimizer)
        dist_optimizer.minimize(avg_loss)

        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())

        status = ExeTrainStatus()
        status.epoch_no = 2
        _, n1 = fleet.save_checkpoint(
            exe, dir_path, trainer_id=0, train_status=status, fs=fs
        )

        status2 = ExeTrainStatus()
        fleet.load_checkpoint(
            exe, dir_path, trainer_id=0, fs=fs, train_status=status2
        )
        self.assertEqual(status2, status)

        _, n2 = fleet.save_checkpoint(
            exe,
            dir_path,
            trainer_id=0,
            train_status=status,
            fs=fs,
            remain_all_checkpoint=False,
        )
        self.assertEqual(n2, n1 + 1)

        c = CheckpointSaver(fs)
        cp_nos = c.get_checkpoint_no(dir_path)
        assert len(cp_nos) == 1  # cleanup all others

        # unnormal
        # test remain_all_checkpoint
        fleet.save_checkpoint(
            exe,
            dir_path,
            trainer_id=0,
            train_status=status,
            fs=fs,
            remain_all_checkpoint=False,
        )

        # can't save under a file
        fs = LocalFS()
        cache_path = "./.load_cache"
        fs.touch(cache_path)
        try:
            fleet.save_checkpoint(
                exe,
                dir_path,
                trainer_id=0,
                train_status=status,
                fs=fs,
                cache_path=cache_path,
            )
            self.assertFalse(True)
        except:
            pass

        # can't load under a file
        try:
            fleet.load_checkpoint(
                exe,
                dir_path,
                trainer_id=0,
                train_status=status2,
                fs=fs,
                cache_path=cache_path,
            )
            self.assertFalse(True)
        except:
            pass
        fs.delete(cache_path)

    def test_hdfs_checkpoint(self):
        fs = HDFSClient("/usr/local/hadoop-2.7.7", None)
        dir_path = "./checkpoint_test_hdfs"
        self._test_checkpoint(fs, os.path.abspath(dir_path))

    def test_local_checkpoint(self):
        fs = LocalFS()
        dir_path = "./checkpoint_test_local"
        self._test_checkpoint(fs, dir_path)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
