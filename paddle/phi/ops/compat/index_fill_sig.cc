
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

KernelSignature IndexFillOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_fill",
                             {"X"},
                             {"IndexTensor", "AxisTensor", "fill_value"},
                             {"Out"});
    } else {
      return KernelSignature(
          "index_fill", {"X"}, {"index", "AxisTensor", "fill_value"}, {"Out"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature(
          "index_fill", {"X"}, {"IndexTensor", "axis", "fill_value"}, {"Out"});
    } else {
      return KernelSignature(
          "index_fill", {"X"}, {"index", "axis", "fill_value"}, {"Out"});
    }
  }
}

KernelSignature IndexFillGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_fill_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "AxisTensor", "fill_value"},
                             {"X@GRAD"});
    } else {
      return KernelSignature("index_fill_grad",
                             {"Out@GRAD"},
                             {"index", "AxisTensor", "fill_value"},
                             {"X@GRAD"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_fill_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "axis", "fill_value"},
                             {"X@GRAD"});
    } else {
      return KernelSignature("index_fill_grad",
                             {"Out@GRAD"},
                             {"index", "axis", "fill_value"},
                             {"X@GRAD"});
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(index_fill, phi::IndexFillOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(index_fill_grad,
                           phi::IndexFillGradOpArgumentMapping);
