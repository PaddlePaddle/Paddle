// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature AssignValueOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  // Here we must use `dtype` attr to determine which attr to use, we can't
  // judge by whether the attr is empty, some unittests will failed
  int dtype = paddle::any_cast<int>(ctx.Attr("dtype"));
  // heer we can't depend on the fluid proto::VarType, so we use the dtype enum
  // value directly, If the enum value is updated, the code also needs to be
  // updated here, but the probability of updating the enum value is very low
  if (dtype == /*BOOL*/ 0) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "bool_values"}, {"Out"});
  } else if (dtype == /*INT32*/ 2) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "int32_values"}, {"Out"});
  } else if (dtype == /*FP32*/ 5) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "fp32_values"}, {"Out"});
  } else if (dtype == /*INT64*/ 3) {
    return KernelSignature(
        "assign_value", {}, {"shape", "dtype", "int64_values"}, {"Out"});
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(assign_value, phi::AssignValueOpArgumentMapping);
