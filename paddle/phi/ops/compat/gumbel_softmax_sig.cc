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

KernelSignature GumbelSoftmaxOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  bool is_test = false;
  if (ctx.HasAttr("is_test")) {
    is_test = paddle::any_cast<bool>(ctx.Attr("is_test"));
  }
  if (is_test) {
    return KernelSignature("gumbel_softmax_infer",
                           {"X"},
                           {"temperature", "hard", "axis"},
                           {"Out"});
  } else {
    return KernelSignature(
        "gumbel_softmax", {"X"}, {"temperature", "hard", "axis"}, {"Out"});
  }
}

KernelSignature GumbelSoftmaxGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "gumbel_softmax_grad", {"Out", "Out@GRAD"}, {"axis"}, {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(gumbel_softmax, phi::GumbelSoftmaxOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(gumbel_softmax_grad,
                           phi::GumbelSoftmaxGradOpArgumentMapping);
