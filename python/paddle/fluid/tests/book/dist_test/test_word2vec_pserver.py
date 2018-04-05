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

import os
import test_word2vec

os.environ["GLOG_v"] = '3'
os.environ["GLOG_logtostderr"] = '1'

os.environ["PADDLE_INIT_PSERVERS"] = "127.0.0.1"
os.environ["TRAINERS"] = "1"
os.environ["POD_IP"] = "127.0.0.1"
os.environ["PADDLE_INIT_TRAINER_ID"] = "0"

os.environ["TRAINING_ROLE"] = "PSERVER"

test_word2vec.train()
