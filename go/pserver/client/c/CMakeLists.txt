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
cc_library(paddle_go_optimizer DEPS paddle_optimizer paddle_proto glog gflags protobuf)
target_link_libraries(paddle_go_optimizer stdc++ m)

# Copy library to the required place.
# See: go/pserver/optimizer.go:
# // #cgo LDFLAGS: ${SRCDIR}/client/c/libpaddle_go_optimizer.a -lstdc++ -lm
add_custom_command(TARGET paddle_go_optimizer POST_BUILD
  COMMAND cp "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_go_optimizer.a" "${CMAKE_CURRENT_SOURCE_DIR}"
  )

go_library(paddle_pserver_cclient STATIC DEPS paddle_go_optimizer)
if(WITH_TESTING)
  # FIXME: this test requires pserver which is not managed by the test
  # we need some kind of e2e testing machanism.
  # add_subdirectory(test)
endif()
