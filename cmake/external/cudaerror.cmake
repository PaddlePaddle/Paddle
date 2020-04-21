# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# Funciton to Download the dependencies during compilation
# This function has 2 parameters, URL / DIRNAME:
# 1. URL:           The download url of 3rd dependencies
# 2. NAME:          The name of file, that determin the dirname
#
FUNCTION(file_download_and_uncompress URL NAME)
  MESSAGE(STATUS "Download dependence[${NAME}] from ${URL}")
  SET(${NAME}_INCLUDE_DIR ${THIRD_PARTY_PATH}/${NAME}/data PARENT_SCOPE)
  ExternalProject_Add(
      extern_download_${NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${THIRD_PARTY_PATH}/${NAME}
      URL                   ${URL}
      DOWNLOAD_DIR          ${THIRD_PARTY_PATH}/${NAME}/data/
      SOURCE_DIR            ${THIRD_PARTY_PATH}/${NAME}/data/
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
    )
ENDFUNCTION()

# download file
set(CUDAERROR_URL  "http://paddlepaddledeps.bj.bcebos.com/cudaErrorMessage.tar.gz" CACHE STRING "" FORCE)
file_download_and_uncompress(${CUDAERROR_URL} "cudaerror")
list(APPEND third_party_deps extern_download_cudaerror)
