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
#

# fluid_modules exclude API-interface of inference/api and inference/capi_exp
get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
get_property(phi_modules GLOBAL PROPERTY PHI_MODULES)
set(utils_modules stringpiece pretty_log string_helper)

set(C_API_SRCS pd_config.cc pd_predictor.cc pd_tensor.cc pd_utils.cc)

# shared inference library deps
set(SHARED_INFERENCE_DEPS ${fluid_modules} ${phi_modules} analysis_predictor)

if (WITH_CRYPTO)
    set(SHARED_INFERENCE_DEPS ${SHARED_INFERENCE_DEPS} paddle_crypto)
endif (WITH_CRYPTO)

if (WITH_PSCORE)
    set(SHARED_INFERENCE_DEPS ${SHARED_INFERENCE_DEPS} fleet ps_service tensor_table)
endif ()

if (WITH_ONNXRUNTIME)
  set(SHARED_INFERENCE_DEPS ${SHARED_INFERENCE_DEPS} onnxruntime_predictor)
endif (WITH_ONNXRUNTIME)

cc_library(paddle_inference_c SRCS ${C_API_SRCS} DEPS paddle_inference ${fluid_modules} ${phi_modules} ${utils_modules})

# Create inference capi shared library
cc_library(paddle_inference_c_shared SHARED SRCS ${C_API_SRCS} DEPS paddle_inference ${SHARED_INFERENCE_DEPS})
set_target_properties(paddle_inference_c_shared PROPERTIES OUTPUT_NAME paddle_inference_c)
if(WIN32)
    target_link_libraries(paddle_inference_c_shared shlwapi.lib)
endif()
