# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
#

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif(APPLE)

cc_library(tape_variable SRCS variable.cc DEPS ${FLUID_CORE_MODULES})
cc_library(tape SRCS tape.cc DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB} tape_variable)

cc_test(test_tape
        SRCS test_tape.cc
        DEPS tape tape_variable)
