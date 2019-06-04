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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import six

__version__ = '0.1'

#assert six.PY3, 'atarashi work only in python3'

log = logging.getLogger('atarashi')
console = logging.StreamHandler()
log.addHandler(console)
#file_handler = logging.FileHandler("./log")
#from colorlog import ColoredFormatter
formatter = logging.Formatter(
    fmt='[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s')
console.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.propagate = False

from paddle.fluid.incubate.atarashi.train import *
from paddle.fluid.incubate.atarashi.types import *
from paddle.fluid.incubate.atarashi.util import ArgumentParser, parse_hparam, parse_runconfig
