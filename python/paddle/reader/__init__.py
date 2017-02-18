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

# It would be too lengthy to require our users to prefix decorators with `decorator`.
# For example, we want the following line
#
#     r = paddle.reader.decorator.bufferd(paddle.reader.creator.text("hello.txt"))
#
# to be a shorter version:
#
#     r = paddle.reader.buffered(paddle.reader.creator.text("hello.txt"))
from decorator import *
