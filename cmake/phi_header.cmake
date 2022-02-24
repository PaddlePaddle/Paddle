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

set(PADDLE_INFERENCE_INSTALL_DIR "${CMAKE_BINARY_DIR}/paddle_inference_install_dir")

function(phi_header_path_compat TARGET_PATH)
message(STATUS "phi header path compat processing: ${TARGET_PATH}")
string(FIND ${TARGET_PATH} "experimental" pos)
if (pos GREATER 1)
    file(GLOB HEADERS "${TARGET_PATH}/*" "*.h")
    foreach(header ${HEADERS})
        if (${header} MATCHES ".*.h$")
            file(READ ${header} HEADER_CONTENT)
            string(REPLACE "paddle/phi/" "paddle/include/experimental/phi/" HEADER_CONTENT "${HEADER_CONTENT}")
            string(REPLACE "paddle/utils/" "paddle/include/experimental/utils/" HEADER_CONTENT "${HEADER_CONTENT}")
            file(WRITE ${header} "${HEADER_CONTENT}")
            message(STATUS "phi header path compat processing complete: ${header}")
        endif()
    endforeach()
endif()
endfunction()

phi_header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental)
phi_header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/phi/api)
phi_header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/phi/api/ext)
phi_header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/phi/api/include)
phi_header_path_compat(${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/phi/common)

# In order to be compatible with the original behavior, the header file name needs to be changed
file(RENAME ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/extension.h
            ${PADDLE_INFERENCE_INSTALL_DIR}/paddle/include/experimental/ext_all.h)