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

KernelSignature IndexAddOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add",
                             {"X"},
                             {"IndexTensor", "AxisTensor", "add_value"},
                             {"Out"});
    } else {
      return KernelSignature(
          "index_add", {"X"}, {"index", "AxisTensor", "add_value"}, {"Out"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature(
          "index_add", {"X"}, {"IndexTensor", "axis", "add_value"}, {"Out"});
    } else {
      return KernelSignature(
          "index_add", {"X"}, {"index", "axis", "add_value"}, {"Out"});
    }
  }
}

KernelSignature IndexAddGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "AxisTensor", "add_value"},
                             {"X@GRAD"});
    } else {
      return KernelSignature("index_add_grad",
                             {"Out@GRAD"},
                             {"index", "AxisTensor", "add_value"},
                             {"X@GRAD"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "axis", "add_value"},
                             {"X@GRAD"});
    } else {
      return KernelSignature("index_add_grad",
                             {"Out@GRAD"},
                             {"index", "axis", "add_value"},
                             {"X@GRAD"});
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(index_add, phi::IndexAddOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(index_add_grad,
                           phi::IndexAddGradOpArgumentMapping);