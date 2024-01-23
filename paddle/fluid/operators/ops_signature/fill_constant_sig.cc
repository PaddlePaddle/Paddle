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

// we have to return every specific KernelSignature for infrt now
KernelSignature FillConstantOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorOutput("Out")) {
    if (ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full", {}, {"ShapeTensor", "ValueTensor", "dtype"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full", {}, {"ShapeTensor", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature(
              "full", {}, {"ShapeTensor", "str_value", "dtype"}, {"Out"});
        }
      }
    } else if (ctx.InputSize("ShapeTensorList") > 0) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full", {}, {"ShapeTensorList", "ValueTensor", "dtype"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full", {}, {"ShapeTensorList", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature(
              "full", {}, {"ShapeTensorList", "str_value", "dtype"}, {"Out"});
        }
      }
    } else {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full", {}, {"shape", "ValueTensor", "dtype"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full", {}, {"shape", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature(
              "full", {}, {"shape", "str_value", "dtype"}, {"Out"});
        }
      }
    }
  } else if (ctx.IsSelectedRowsOutput("Out")) {
    if (ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full_sr", {}, {"ShapeTensor", "ValueTensor", "dtype"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full_sr", {}, {"ShapeTensor", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature(
              "full_sr", {}, {"ShapeTensor", "str_value", "dtype"}, {"Out"});
        }
      }
    } else if (ctx.InputSize("ShapeTensorList") > 0) {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature("full_sr",
                               {},
                               {"ShapeTensorList", "ValueTensor", "dtype"},
                               {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full_sr", {}, {"ShapeTensorList", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature("full_sr",
                                 {},
                                 {"ShapeTensorList", "str_value", "dtype"},
                                 {"Out"});
        }
      }
    } else {
      if (ctx.HasInput("ValueTensor")) {
        return KernelSignature(
            "full_sr", {}, {"shape", "ValueTensor", "dtype"}, {"Out"});
      } else {
        const auto& str_value =
            paddle::any_cast<std::string>(ctx.Attr("str_value"));
        if (str_value.empty()) {
          return KernelSignature(
              "full_sr", {}, {"shape", "value", "dtype"}, {"Out"});
        } else {
          return KernelSignature(
              "full_sr", {}, {"shape", "str_value", "dtype"}, {"Out"});
        }
      }
    }
  }
  return KernelSignature("unregistered", {}, {}, {});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(fill_constant, full);

PD_REGISTER_ARG_MAPPING_FN(fill_constant, phi::FillConstantOpArgumentMapping);
