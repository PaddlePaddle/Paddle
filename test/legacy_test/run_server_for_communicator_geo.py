# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("../deprecated/legacy_test")
from test_communicator_geo_deprecated import TestCommunicatorGeoEnd2End

import paddle

paddle.enable_static()

pipe_name = os.getenv("PIPE_FILE")


class RunServer(TestCommunicatorGeoEnd2End):
    def runTest(self):
        pass


os.environ["TRAINING_ROLE"] = "PSERVER"

half_run_server = RunServer()
with open(pipe_name, 'w') as pipe:
    pipe.write('done')

half_run_server.run_ut()
