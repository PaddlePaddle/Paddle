#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from hapi.text.bert.utils.args import str2bool as str2bool
from hapi.text.bert.utils.args import ArgumentGroup as ArgumentGroup
from hapi.text.bert.utils.args import print_arguments as print_arguments
from hapi.text.bert.utils.args import check_cuda as check_cuda

from hapi.text.bert.utils.cards import get_cards as get_cards

from hapi.text.bert.utils.fp16 import cast_fp16_to_fp32 as cast_fp16_to_fp32
from hapi.text.bert.utils.fp16 import cast_fp32_to_fp16 as cast_fp32_to_fp16
from hapi.text.bert.utils.fp16 import copy_to_master_param as copy_to_master_param
from hapi.text.bert.utils.fp16 import create_master_params_grads as create_master_params_grads
from hapi.text.bert.utils.fp16 import master_param_to_train_param as master_param_to_train_param

from hapi.text.bert.utils.init import init_checkpoint as init_checkpoint
from hapi.text.bert.utils.init import init_pretraining_params as init_pretraining_params
from hapi.text.bert.utils.init import init_from_static_model as init_from_static_model
