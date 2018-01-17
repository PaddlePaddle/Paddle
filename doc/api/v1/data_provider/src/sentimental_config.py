#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from paddle.trainer_config_helpers import *

dictionary = dict()
...  #  read dictionary from outside

define_py_data_sources2(
    train_list='train.list',
    test_list=None,
    module='sentimental_provider',
    obj='process',
    # above codes same as mnist sample.
    args={  # pass to provider.
        'dictionary': dictionary
    })
