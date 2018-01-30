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

from __future__ import print_function
import unittest
import os
import paddle.v2.fluid as fluid


class FluidTestRunner(object):
    def __init__(self):
        self.runner = unittest.TextTestRunner(verbosity=10)
        self.suite = unittest.TestSuite()

    def add_test(self, test):
        if hasattr(test, '__iter__'):
            # recursively add test_methods
            for each_test in test:
                self.add_test(each_test)
        else:
            # to each test method, create new Programs and Scope
            previous_run = test.run

            def __hooked_run__(*args, **kwargs):
                prog = fluid.Program()
                startup_prog = fluid.Program()
                scope = fluid.core.Scope()
                with fluid.scope_guard(scope):
                    with fluid.program_guard(prog, startup_prog):
                        previous_run(*args, **kwargs)

            test.run = __hooked_run__
            self.suite.addTest(test)

    def run(self):
        self.runner.run(self.suite)


def main():
    test_loader = unittest.TestLoader()
    runner = FluidTestRunner()
    for test in test_loader.discover(
            start_dir=os.path.dirname(os.path.abspath(__file__)),
            pattern='test_*'):
        runner.add_test(test)

    runner.run()


if __name__ == '__main__':
    main()
