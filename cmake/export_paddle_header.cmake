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

set(PADDLE_INFERENCE_INSTALL_DIR
    "${CMAKE_BINARY_DIR}/paddle_inference_install_dir")

function(header_path_compat TARGET_PATH)
  message(STATUS "header path compat processing: ${TARGET_PATH}")
  file(GLOB HEADERS "${TARGET_PATH}/*" "*.h")
  foreach(header ${HEADERS})
    if(${header} MATCHES ".*.h$")
      file(READ ${header} HEADER_CONTENT)
      string(REPLACE "paddle/fluid/platform/" "paddle/phi/" HEADER_CONTENT
                     "${HEADER_CONTENT}")
      string(REPLACE "paddle/pir/include/" "paddle/pir/" HEADER_CONTENT
                     "${HEADER_CONTENT}")
      string(REPLACE "paddle/fluid/pir/drr/include/" "paddle/pir/drr/"
                     HEADER_CONTENT "${HEADER_CONTENT}")
      string(REPLACE "paddle/fluid/pir/utils/" "paddle/pir/utils/"
                     HEADER_CONTENT "${HEADER_CONTENT}")
      file(WRITE ${header} "${HEADER_CONTENT}")
      message(STATUS "header path compat processing complete: ${header}")
    endif()
  endforeach()
endfunction()

header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle)
header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api/ext)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/api/include)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/common)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/phi/core)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/core)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/core/parser)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/control_flow/ir
)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/shape/ir)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/dialect/shape/utils)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/drr)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/pass)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/pattern_rewrite)
header_path_compat(
  ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/pir/utils)

# NOTE(liuyuanle): In inference lib, no need include paddle/utils/pybind.h, so we delete this.
file(READ ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/extension.h
     HEADER_CONTENT)
string(REGEX REPLACE "#if !defined\\(PADDLE_ON_INFERENCE\\).*#endif" ""
                     HEADER_CONTENT "${HEADER_CONTENT}")
file(WRITE ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/paddle/extension.h
     "${HEADER_CONTENT}")
