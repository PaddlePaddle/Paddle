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

KernelSignature UniformRandomOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  int diag_num = paddle::any_cast<int>(ctx.Attr("diag_num"));
  if (ctx.IsDenseTensorOutput("Out")) {
    if (diag_num) {
      if (ctx.InputSize("ShapeTensorList") > 0) {
<<<<<<< HEAD
        return KernelSignature("uniform_raw",
=======
        return KernelSignature("uniform_random_raw",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                               {},
                               {"ShapeTensorList",
                                "dtype",
                                "min",
                                "max",
                                "seed",
                                "diag_num",
                                "diag_step",
                                "diag_val"},
                               {"Out"});
      } else {
        const auto& shape =
            paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
        if (ctx.HasInput("ShapeTensor") && shape.empty()) {
<<<<<<< HEAD
          return KernelSignature("uniform_raw",
=======
          return KernelSignature("uniform_random_raw",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"ShapeTensor",
                                  "dtype",
                                  "min",
                                  "max",
                                  "seed",
                                  "diag_num",
                                  "diag_step",
                                  "diag_val"},
                                 {"Out"});
        } else {
<<<<<<< HEAD
          return KernelSignature("uniform_raw",
=======
          return KernelSignature("uniform_random_raw",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"shape",
                                  "dtype",
                                  "min",
                                  "max",
                                  "seed",
                                  "diag_num",
                                  "diag_step",
                                  "diag_val"},
                                 {"Out"});
        }
      }
    } else {
      if (ctx.InputSize("ShapeTensorList") > 0) {
        return KernelSignature(
<<<<<<< HEAD
            "uniform",
=======
            "uniform_random",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            {},
            {"ShapeTensorList", "dtype", "min", "max", "seed"},
            {"Out"});
      } else {
        const auto& shape =
            paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
        if (ctx.HasInput("ShapeTensor") && shape.empty()) {
<<<<<<< HEAD
          return KernelSignature("uniform",
=======
          return KernelSignature("uniform_random",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"ShapeTensor", "dtype", "min", "max", "seed"},
                                 {"Out"});
        } else {
<<<<<<< HEAD
          return KernelSignature(
              "uniform", {}, {"shape", "dtype", "min", "max", "seed"}, {"Out"});
=======
          return KernelSignature("uniform_random",
                                 {},
                                 {"shape", "dtype", "min", "max", "seed"},
                                 {"Out"});
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
      }
    }
  } else if (ctx.IsSelectedRowsOutput("Out")) {
    if (diag_num) {
      if (ctx.InputSize("ShapeTensorList") > 0) {
<<<<<<< HEAD
        return KernelSignature("uniform_raw_sr",
=======
        return KernelSignature("uniform_random_raw_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                               {},
                               {"ShapeTensorList",
                                "dtype",
                                "min",
                                "max",
                                "seed",
                                "diag_num",
                                "diag_step",
                                "diag_val"},
                               {"Out"});
      } else {
        const auto& shape =
            paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
        if (ctx.HasInput("ShapeTensor") && shape.empty()) {
<<<<<<< HEAD
          return KernelSignature("uniform_raw_sr",
=======
          return KernelSignature("uniform_random_raw_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"ShapeTensor",
                                  "dtype",
                                  "min",
                                  "max",
                                  "seed",
                                  "diag_num",
                                  "diag_step",
                                  "diag_val"},
                                 {"Out"});
        } else {
<<<<<<< HEAD
          return KernelSignature("uniform_raw_sr",
=======
          return KernelSignature("uniform_random_raw_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"shape",
                                  "dtype",
                                  "min",
                                  "max",
                                  "seed",
                                  "diag_num",
                                  "diag_step",
                                  "diag_val"},
                                 {"Out"});
        }
      }
    } else {
      if (ctx.InputSize("ShapeTensorList") > 0) {
        return KernelSignature(
<<<<<<< HEAD
            "uniform_sr",
=======
            "uniform_random_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            {},
            {"ShapeTensorList", "dtype", "min", "max", "seed"},
            {"Out"});
      } else {
        const auto& shape =
            paddle::any_cast<std::vector<int64_t>>(ctx.Attr("shape"));
        if (ctx.HasInput("ShapeTensor") && shape.empty()) {
<<<<<<< HEAD
          return KernelSignature("uniform_sr",
=======
          return KernelSignature("uniform_random_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"ShapeTensor", "dtype", "min", "max", "seed"},
                                 {"Out"});
        } else {
<<<<<<< HEAD
          return KernelSignature("uniform_sr",
=======
          return KernelSignature("uniform_random_sr",
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                 {},
                                 {"shape", "dtype", "min", "max", "seed"},
                                 {"Out"});
        }
      }
    }
  }
  return KernelSignature("unregistered", {}, {}, {});
}
}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_BASE_KERNEL_NAME(uniform_random, uniform);

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
PD_REGISTER_ARG_MAPPING_FN(uniform_random, phi::UniformRandomOpArgumentMapping);
