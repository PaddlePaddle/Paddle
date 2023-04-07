# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#NOTE: Logic is from
# https://github.com/mindspore-ai/graphengine/blob/master/CMakeLists.txt
if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS
   ${ASCEND_DIR}/ascend-toolkit/latest/fwkacllib/include/graph/ascend_string.h)
  # It means CANN 20.2 +
  add_definitions(-DPADDLE_WITH_ASCEND_STRING)
endif()
