# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SET(PADDLE_BFLOAT_COMMIT_HASH 58ccd641c0df5b4b74d50201ac9a08ec603e8f01)

file(DOWNLOAD
    https://raw.githubusercontent.com/jakpiase/paddle_bfloat/${PADDLE_BFLOAT_COMMIT_HASH}/bfloat16.cc
    ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/paddle_bfloat/bfloat16.cc)

file(DOWNLOAD
    https://raw.githubusercontent.com/jakpiase/paddle_bfloat/${PADDLE_BFLOAT_COMMIT_HASH}/bfloat16.h
    ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/paddle_bfloat/bfloat16.h)
