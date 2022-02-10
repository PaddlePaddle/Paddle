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

#include "paddle/pten/core/compat/op_utils.h"

namespace pten {

// we have to return every specific KernelSignature for infrt now
KernelSignature FillConstantOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput("Out")) {
    if (ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full", {}, {"ShapeTensor", "ValueTensor"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature("full", {}, {"ShapeTensor", "value"}, {"Out"});
        } else {
          return KernelSignature(
              "full", {}, {"ShapeTensor", "str_value"}, {"Out"});
        }
      }
    } else if (ctx.InputSize("ShapeTensorList") > 0) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full", {}, {"ShapeTensorList", "ValueTensor"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full", {}, {"ShapeTensorList", "value"}, {"Out"});
        } else {
          return KernelSignature(
              "full", {}, {"ShapeTensorList", "str_value"}, {"Out"});
        }
      }
    } else {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature("full", {}, {"shape", "ValueTensor"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature("full", {}, {"shape", "value"}, {"Out"});
        } else {
          return KernelSignature("full", {}, {"shape", "str_value"}, {"Out"});
        }
      }
    }
  }
  return KernelSignature("unregistered", {}, {}, {});
}

}  // namespace pten

PT_REGISTER_BASE_KERNEL_NAME(fill_constant, full);

PT_REGISTER_ARG_MAPPING_FN(fill_constant, pten::FillConstantOpArgumentMapping);
