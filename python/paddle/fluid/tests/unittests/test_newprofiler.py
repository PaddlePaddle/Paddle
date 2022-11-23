#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import tempfile
import os
import paddle
import paddle.profiler as profiler
import paddle.profiler.utils as utils
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader


class TestProfiler(unittest.TestCase):

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_profiler(self):

        def my_trace_back(prof):
            path = os.path.join(self.temp_dir.name,
                                './test_profiler_chrometracing')
            profiler.export_chrome_tracing(path)(prof)
            path = os.path.join(self.temp_dir.name, './test_profiler_pb')
            profiler.export_protobuf(path)(prof)

        self.temp_dir = tempfile.TemporaryDirectory()
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value,
                             stop_gradient=False,
                             place=paddle.CPUPlace())
        y = x / 2.0
        ones_like_y = paddle.ones_like(y)
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU], ) as prof:
            y = x / 2.0
        prof = None
        self.assertEqual(utils._is_profiler_used, False)
        with profiler.RecordEvent(name='test'):
            y = x / 2.0

        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=(1, 2)) as prof:
            self.assertEqual(utils._is_profiler_used, True)
            with profiler.RecordEvent(name='test'):
                y = x / 2.0

        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=profiler.make_scheduler(closed=0,
                                                                 ready=1,
                                                                 record=1,
                                                                 repeat=1),
                               on_trace_ready=my_trace_back) as prof:
            y = x / 2.0
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=profiler.make_scheduler(closed=0,
                                                                 ready=0,
                                                                 record=2,
                                                                 repeat=1),
                               on_trace_ready=my_trace_back) as prof:
            for i in range(3):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN,
                on_trace_ready=my_trace_back) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()

        def my_sheduler(num_step):
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD_AND_RETURN
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED

        def my_sheduler1(num_step):
            if num_step % 5 < 2:
                return profiler.ProfilerState.RECORD
            elif num_step % 5 < 3:
                return profiler.ProfilerState.READY
            elif num_step % 5 < 4:
                return profiler.ProfilerState.RECORD
            else:
                return profiler.ProfilerState.CLOSED

        prof = None
        with profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU],
                scheduler=lambda x: profiler.ProfilerState.RECORD_AND_RETURN,
                on_trace_ready=my_trace_back) as prof:
            for i in range(2):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=my_sheduler,
                               on_trace_ready=my_trace_back) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=my_sheduler1) as prof:
            for i in range(5):
                y = x / 2.0
                prof.step()
        prof = None
        with profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
                               scheduler=profiler.make_scheduler(closed=1,
                                                                 ready=1,
                                                                 record=2,
                                                                 repeat=1,
                                                                 skip_first=1),
                               on_trace_ready=my_trace_back,
                               profile_memory=True,
                               record_shapes=True) as prof:
            for i in range(5):
                y = x / 2.0
                paddle.grad(outputs=y, inputs=[x], grad_outputs=ones_like_y)
                prof.step()

        path = os.path.join(self.temp_dir.name, './test_profiler_pb.pb')
        prof.export(path=path, format='pb')
        prof.summary()
        result = profiler.utils.load_profiler_result(path)
        prof = None
        dataset = RandomDataset(10 * 4)
        simple_net = SimpleNet()
        opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                   parameters=simple_net.parameters())
        loader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            drop_last=True,
                            num_workers=2)
        prof = profiler.Profiler(on_trace_ready=lambda prof: None)
        prof.start()
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.minimize(avg_loss)
            simple_net.clear_gradients()
            prof.step()
        prof.stop()
        prof.summary()
        prof = None
        dataset = RandomDataset(10 * 4)
        simple_net = SimpleNet()
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
        opt = paddle.optimizer.Adam(learning_rate=1e-3,
                                    parameters=simple_net.parameters())
        prof = profiler.Profiler(on_trace_ready=lambda prof: None)
        prof.start()
        for i, (image, label) in enumerate(loader()):
            out = simple_net(image)
            loss = F.cross_entropy(out, label)
            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.step()
            simple_net.clear_gradients()
            prof.step()
        prof.stop()


class TestNvprof(unittest.TestCase):

    def test_nvprof(self):
        for i in range(10):
            paddle.fluid.profiler._nvprof_range(i, 10, 20)
            x_value = np.random.randn(2, 3, 3)
            x = paddle.to_tensor(x_value,
                                 stop_gradient=False,
                                 place=paddle.CPUPlace())
            y = x / 2.0


class TestGetProfiler(unittest.TestCase):

    def test_getprofiler(self):
        config_content = '''
        {
        "targets": ["CPU"],
        "scheduler": [3,4],
        "on_trace_ready": {
            "export_chrome_tracing":{
                "module": "paddle.profiler",
                "use_direct": false,
                "args": [],
                "kwargs": {
                        "dir_name": "testdebug/"
                    }
                }
            },
          "timer_only": false
        }
        '''
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        import paddle.profiler.profiler as profiler
        profiler = profiler.get_profiler(filehandle.name)
        x_value = np.random.randn(2, 3, 3)
        x = paddle.to_tensor(x_value,
                             stop_gradient=False,
                             place=paddle.CPUPlace())
        with profiler:
            for i in range(5):
                y = x / 2.0
                ones_like_y = paddle.ones_like(y)
                profiler.step()

        # below tests are just for coverage, wrong config
        # test use_direct
        config_content = '''
        {
        "targets": ["Cpu", "Gpu"],
        "scheduler": {
            "make_scheduler":{
                "module": "paddle.profiler",
                "use_direct": true,
                "args": [],
                "kwargs": {}
            }
        },
        "on_trace_ready": {
            "export_chrome_tracing":{
                "module": "paddle.profiler1",
                "use_direct": true,
                "args": [],
                "kwargs": {
                    }
                }
            },
          "timer_only": false
        }
        '''
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        import paddle.profiler.profiler as profiler
        try:
            profiler = profiler.get_profiler(filehandle.name)
        except:
            pass

        # test scheduler
        config_content = '''
        {
        "targets": ["Cpu", "Gpu"],
        "scheduler": {
           "make_scheduler":{
                "module": "paddle.profiler",
                "use_direct": false,
                "args": [],
                "kwargs": {
                        "closed": 1,
                        "ready": 1,
                        "record": 2
                    }
            }
        },
        "on_trace_ready": {
            "export_chrome_tracing":{
                "module": "paddle.profiler",
                "use_direct": true,
                "args": [],
                "kwargs": {
                    }
                }
            },
          "timer_only": false
        }
        '''
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        import paddle.profiler.profiler as profiler
        profiler = profiler.get_profiler(filehandle.name)

        # test exception
        config_content = '''
        {
        "targets": [1],
        "scheduler": {
            "make_scheduler1":{
                "module": "paddle.profiler",
                "use_direct": false,
                "args": [],
                "kwargs": {
                        "closed": 1,
                        "ready": 1,
                        "record": 2
                    }
            }
        },
        "on_trace_ready": {
            "export_chrome_tracing1":{
                "module": "paddle.profiler",
                "use_direct": false,
                "args": [],
                "kwargs": {
                        "dir_name": "testdebug/"
                    }
                }
            },
          "timer_only": 1
        }
        '''
        filehandle = tempfile.NamedTemporaryFile(mode='w')
        filehandle.write(config_content)
        filehandle.flush()
        import paddle.profiler.profiler as profiler
        profiler = profiler.get_profiler(filehandle.name)
        # test path error
        import paddle.profiler.profiler as profiler
        profiler = profiler.get_profiler('nopath.json')


class RandomDataset(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([100]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class SimpleNet(nn.Layer):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, image, label=None):
        return self.fc(image)


class TestTimerOnly(unittest.TestCase):

    def test_with_dataloader(self):

        def train(step_num_samples=None):
            dataset = RandomDataset(20 * 4)
            simple_net = SimpleNet()
            opt = paddle.optimizer.SGD(learning_rate=1e-3,
                                       parameters=simple_net.parameters())
            loader = DataLoader(dataset,
                                batch_size=4,
                                shuffle=True,
                                drop_last=True,
                                num_workers=2)
            step_info = ''
            p = profiler.Profiler(timer_only=True)
            p.start()
            for i, (image, label) in enumerate(loader()):
                out = simple_net(image)
                loss = F.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                simple_net.clear_gradients()
                p.step(num_samples=step_num_samples)
                if i % 10 == 0:
                    step_info = p.step_info()
                    print("Iter {}: {}".format(i, step_info))
            p.stop()
            return step_info

        step_info = train(step_num_samples=None)
        self.assertTrue('steps/s' in step_info)
        step_info = train(step_num_samples=4)
        self.assertTrue('samples/s' in step_info)

    def test_without_dataloader(self):
        x = paddle.to_tensor(np.random.randn(10, 10))
        y = paddle.to_tensor(np.random.randn(10, 10))
        p = profiler.Profiler(timer_only=True)
        p.start()
        step_info = ''
        for i in range(20):
            out = x + y
            p.step()
        p.stop()


if __name__ == '__main__':
    unittest.main()
