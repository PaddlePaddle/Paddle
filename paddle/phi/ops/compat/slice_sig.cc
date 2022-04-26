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

KernelSignature SliceOpArgumentMapping(const ArgumentMappingContext& ctx) {
  // if input is Tensor Array
  if (ctx.IsDenseTensorVectorInput("Input")) {
    return KernelSignature("unregistered", {}, {}, {});
  }

  if (ctx.HasInput("StartsTensor")) {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature("slice",
                             {"Input"},
                             {"axes",
                              "StartsTensor",
                              "EndsTensor",
                              "infer_flags",
                              "decrease_axis"},
                             {"Out"});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature("slice",
                             {"Input"},
                             {"axes",
                              "StartsTensor",
                              "EndsTensorList",
                              "infer_flags",
                              "decrease_axis"},
                             {"Out"});
    } else {
      return KernelSignature(
          "slice",
          {"Input"},
          {"axes", "StartsTensor", "ends", "infer_flags", "decrease_axis"},
          {"Out"});
    }
  } else if (ctx.InputSize("StartsTensorList") > 0) {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature("slice",
                             {"Input"},
                             {"axes",
                              "StartsTensorList",
                              "EndsTensor",
                              "infer_flags",
                              "decrease_axis"},
                             {"Out"});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature("slice",
                             {"Input"},
                             {"axes",
                              "StartsTensorList",
                              "EndsTensorList",
                              "infer_flags",
                              "decrease_axis"},
                             {"Out"});
    } else {
      return KernelSignature(
          "slice",
          {"Input"},
          {"axes", "StartsTensorList", "ends", "infer_flags", "decrease_axis"},
          {"Out"});
    }
  } else {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature(
          "slice",
          {"Input"},
          {"axes", "starts", "EndsTensor", "infer_flags", "decrease_axis"},
          {"Out"});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature(
          "slice",
          {"Input"},
          {"axes", "starts", "EndsTensorList", "infer_flags", "decrease_axis"},
          {"Out"});
    } else {
      return KernelSignature(
          "slice",
          {"Input"},
          {"axes", "starts", "ends", "infer_flags", "decrease_axis"},
          {"Out"});
    }
  }
}

KernelSignature SliceGradOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorVectorInput("Input")) {
    return KernelSignature("unregistered", {}, {}, {});
  }

  if (ctx.HasInput("StartsTensor")) {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature("slice_grad",
                             {"Input", GradVarName("Out")},
                             {"axes",
                              "StartsTensor",
                              "EndsTensor",
                              "infer_flags",
                              "decrease_axis"},
                             {GradVarName("Input")});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature("slice_grad",
                             {"Input", GradVarName("Out")},
                             {"axes",
                              "StartsTensor",
                              "EndsTensorList",
                              "infer_flags",
                              "decrease_axis"},
                             {GradVarName("Input")});
    } else {
      return KernelSignature(
          "slice_grad",
          {"Input", GradVarName("Out")},
          {"axes", "StartsTensor", "ends", "infer_flags", "decrease_axis"},
          {GradVarName("Input")});
    }
  } else if (ctx.InputSize("StartsTensorList") > 0) {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature("slice_grad",
                             {"Input", GradVarName("Out")},
                             {"axes",
                              "StartsTensorList",
                              "EndsTensor",
                              "infer_flags",
                              "decrease_axis"},
                             {GradVarName("Input")});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature("slice_grad",
                             {"Input", GradVarName("Out")},
                             {"axes",
                              "StartsTensorList",
                              "EndsTensorList",
                              "infer_flags",
                              "decrease_axis"},
                             {GradVarName("Input")});
    } else {
      return KernelSignature(
          "slice_grad",
          {"Input", GradVarName("Out")},
          {"axes", "StartsTensorList", "ends", "infer_flags", "decrease_axis"},
          {GradVarName("Input")});
    }
  } else {
    if (ctx.HasInput("EndsTensor")) {
      return KernelSignature(
          "slice_grad",
          {"Input", GradVarName("Out")},
          {"axes", "starts", "EndsTensor", "infer_flags", "decrease_axis"},
          {GradVarName("Input")});
    } else if (ctx.InputSize("EndsTensorList") > 0) {
      return KernelSignature(
          "slice_grad",
          {"Input", GradVarName("Out")},
          {"axes", "starts", "EndsTensorList", "infer_flags", "decrease_axis"},
          {GradVarName("Input")});
    } else {
      return KernelSignature(
          "slice_grad",
          {"Input", GradVarName("Out")},
          {"axes", "starts", "ends", "infer_flags", "decrease_axis"},
          {GradVarName("Input")});
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(slice, phi::SliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(slice_grad, phi::SliceGradOpArgumentMapping);
