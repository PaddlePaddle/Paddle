# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import static

paddle.enable_static()


class TestOpPriority(unittest.TestCase):
    def test_op_priority(self):
        # In this test case, x and y share the same data,
        # which is initialized to 0. The shared data is
        # read and wrote by two concurrent Ops increment(x)
        # and increment(y). In case of Op sequential scheduling,
        # the result of increment(x) would be 1 while that of
        # increment(y) would be 2. However, increment(y) is
        # set to a higher priority than increment(x), so the
        # result of increment(y) would be 1.
        program = static.Program()
        with static.program_guard(program):
            x = paddle.zeros(shape=[1], dtype='int32')
            block = program.global_block()

            y = block.create_var(dtype='int32')
            block.append_op(
                type='share_data', inputs={'X': x.name}, outputs={'Out': y.name}
            )

            paddle.increment(x)
            block.ops[-1].dist_attr.scheduling_priority = 1
            paddle.increment(y)
            block.ops[-1].dist_attr.scheduling_priority = -1

            # Note that the priority order involved cross-thread scheduling
            # is not guaranteed in standalone executor. As fetch(y)
            # is scheduled in the different thread from increment(x),
            # they are not scheduled in priority order. To make sure that
            # fetch(y) is scheduled before increment(x) in priority order,
            # we tricky enable serial_run here.
            paddle.framework.set_flags({'FLAGS_new_executor_serial_run': 1})

            exe = static.Executor()
            # Currently, priority scheduling is not supported in the first
            # step that builds Op list by running kernel. Remove the first
            # run here when static-build without kernel running is supported.
            result = exe.run(program, fetch_list=[y])
            result = exe.run(program, fetch_list=[y])
            self.assertEqual(result[0], 1)


if __name__ == "__main__":
    unittest.main()
