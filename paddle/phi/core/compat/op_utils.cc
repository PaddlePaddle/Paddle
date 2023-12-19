/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

DefaultKernelSignatureMap& DefaultKernelSignatureMap::Instance() {
  static DefaultKernelSignatureMap g_default_kernel_sig_map;
  return g_default_kernel_sig_map;
}

OpUtilsMap& OpUtilsMap::Instance() {
  static OpUtilsMap g_op_utils_map;
  return g_op_utils_map;
}

BaseKernelNameRegistrar::BaseKernelNameRegistrar(const char* op_type,
                                                 const char* base_kernel_name) {
  OpUtilsMap::Instance().InsertBaseKernelName(op_type, base_kernel_name);
  OpUtilsMap::Instance().InsertFluidOplName(op_type, base_kernel_name);
}

ArgumentMappingFnRegistrar::ArgumentMappingFnRegistrar(
    const char* op_type, ArgumentMappingFn arg_mapping_fn) {
  OpUtilsMap::Instance().InsertArgumentMappingFn(op_type,
                                                 std::move(arg_mapping_fn));
}

}  // namespace phi
