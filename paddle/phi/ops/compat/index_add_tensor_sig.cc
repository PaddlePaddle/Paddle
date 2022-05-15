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

KernelSignature IndexAddTensorOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_tensor",
                             {"X", "AddValue"},
                             {"IndexTensor", "AxisTensor"},
                             {"Out"});
    } else {
      return KernelSignature("index_add_tensor",
                             {"X", "AddValue"},
                             {"index", "AxisTensor"},
                             {"Out"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_tensor",
                             {"X", "AddValue"},
                             {"IndexTensor", "axis"},
                             {"Out"});
    } else {
      return KernelSignature(
          "index_add_tensor", {"X", "AddValue"}, {"index", "axis"}, {"Out"});
    }
  }
}

KernelSignature IndexAddTensorGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("AxisTensor")) {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_tensor_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "AxisTensor"},
                             {"X@GRAD", "AddValue@GRAD"});
    } else {
      return KernelSignature("index_add_tensor_grad",
                             {"Out@GRAD"},
                             {"index", "AxisTensor"},
                             {"X@GRAD", "AddValue@GRAD"});
    }
  } else {
    if (ctx.HasInput("IndexTensor")) {
      return KernelSignature("index_add_tensor_grad",
                             {"Out@GRAD"},
                             {"IndexTensor", "axis"},
                             {"X@GRAD", "AddValue@GRAD"});
    } else {
      return KernelSignature("index_add_tensor_grad",
                             {"Out@GRAD"},
                             {"index", "axis"},
                             {"X@GRAD", "AddValue@GRAD"});
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(index_add_tensor,
                           phi::IndexAddTensorOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(index_add_tensor_grad,
                           phi::IndexAddTensorGradOpArgumentMapping);