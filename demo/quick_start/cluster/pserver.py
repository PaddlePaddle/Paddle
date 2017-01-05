# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from py_paddle import swig_paddle as api

#import pudb;pudb.set_trace()


def main():
    api.initPaddle("--nics=lo0", "--port=7164", "--ports_num=1",
                   "--num_gradient_servers=1", "--comment=paddle_pserver")
    pserver = api.ParameterServer.createParameterServer()
    pserver.init()
    pserver.start()
    pserver.join()


if __name__ == '__main__':
    main()
