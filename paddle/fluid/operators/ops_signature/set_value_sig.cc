
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

KernelSignature SetValueOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("Input")) {
    if (ctx.InputSize("StartsTensorList") > 0) {
      if (ctx.InputSize("EndsTensorList") > 0) {
        if (ctx.InputSize("StepsTensorList") > 0) {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"StartsTensorList",
                                    "EndsTensorList",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"StartsTensorList",
                                    "EndsTensorList",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        } else {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"StartsTensorList",
                                    "EndsTensorList",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"StartsTensorList",
                                    "EndsTensorList",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        }
      } else {
        if (ctx.InputSize("StepsTensorList") > 0) {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"StartsTensorList",
                                    "ends",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"StartsTensorList",
                                    "ends",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        } else {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"StartsTensorList",
                                    "ends",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"StartsTensorList",
                                    "ends",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        }
      }
    } else {
      if (ctx.InputSize("EndsTensorList") > 0) {
        if (ctx.InputSize("StepsTensorList") > 0) {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"starts",
                                    "EndsTensorList",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"starts",
                                    "EndsTensorList",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        } else {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"starts",
                                    "EndsTensorList",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"starts",
                                    "EndsTensorList",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        }
      } else {
        if (ctx.InputSize("StepsTensorList") > 0) {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"starts",
                                    "ends",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"starts",
                                    "ends",
                                    "StepsTensorList",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        } else {
          if (ctx.HasInput("ValueTensor")) {
            return KernelSignature("set_value_with_tensor",
                                   {"Input", "ValueTensor"},
                                   {"starts",
                                    "ends",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes"},
                                   {"Out"});
          } else {
            return KernelSignature("set_value",
                                   {"Input"},
                                   {"starts",
                                    "ends",
                                    "steps",
                                    "axes",
                                    "decrease_axes",
                                    "none_axes",
                                    "shape",
                                    "values"},
                                   {"Out"});
          }
        }
      }
    }
  }
  return KernelSignature("unregistered", {}, {}, {});
}

KernelSignature SetValueGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.InputSize("StartsTensorList") > 0) {
    if (ctx.InputSize("EndsTensorList") > 0) {
      if (ctx.InputSize("StepsTensorList") > 0) {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"StartsTensorList",
                                "EndsTensorList",
                                "StepsTensorList",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      } else {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"StartsTensorList",
                                "EndsTensorList",
                                "steps",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      }
    } else {
      if (ctx.InputSize("StepsTensorList") > 0) {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"StartsTensorList",
                                "ends",
                                "StepsTensorList",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      } else {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"StartsTensorList",
                                "ends",
                                "steps",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      }
    }
  } else {
    if (ctx.InputSize("EndsTensorList") > 0) {
      if (ctx.InputSize("StepsTensorList") > 0) {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"starts",
                                "EndsTensorList",
                                "StepsTensorList",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      } else {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"starts",
                                "EndsTensorList",
                                "steps",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      }
    } else {
      if (ctx.InputSize("StepsTensorList") > 0) {
        return KernelSignature("set_value_grad",
                               {"Out@GRAD"},
                               {"starts",
                                "ends",
                                "StepsTensorList",
                                "axes",
                                "decrease_axes",
                                "none_axes"},
                               {"Input@GRAD", "ValueTensor@GRAD"});
      } else {
        return KernelSignature(
            "set_value_grad",
            {"Out@GRAD"},
            {"starts", "ends", "steps", "axes", "decrease_axes", "none_axes"},
            {"Input@GRAD", "ValueTensor@GRAD"});
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(set_value, phi::SetValueOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(set_value_grad, phi::SetValueGradOpArgumentMapping);
