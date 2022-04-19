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

#include <string>

#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/utils/small_vector.h"

namespace phi {

KernelSignature StridedSliceOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& starts = paddle::any_cast<std::vector<int>>(ctx.Attr("starts"));
  const auto& ends = paddle::any_cast<std::vector<int>>(ctx.Attr("ends"));
  const auto& strides = paddle::any_cast<std::vector<int>>(ctx.Attr("strides"));

  bool use_attr_starts = !ctx.IsRuntime() && !starts.empty();
  bool use_attr_ends = !ctx.IsRuntime() && !ends.empty();
  bool use_attr_strides = !ctx.IsRuntime() && !strides.empty();

  const char* starts_key =
      ctx.HasInput("StartsTensor")
          ? "StartsTensor"
          : (ctx.InputSize("StartsTensorList") > 0
                 ? (use_attr_starts ? "starts" : "StartsTensorList")
                 : "starts");
  const char* ends_key =
      ctx.HasInput("EndsTensor")
          ? "EndsTensor"
          : (ctx.InputSize("EndsTensorList") > 0
                 ? (use_attr_ends ? "ends" : "EndsTensorList")
                 : "ends");
  const char* strides_key =
      ctx.HasInput("StridesTensor")
          ? "StridesTensor"
          : (ctx.InputSize("StridesTensorList") > 0
                 ? (use_attr_strides ? "strides" : "StridesTensorList")
                 : "strides");

  paddle::SmallVector<const char*> inputs = {"Input"};
  paddle::SmallVector<const char*> attrs = {"axes",
                                            starts_key,
                                            ends_key,
                                            strides_key,
                                            "infer_flags",
                                            "decrease_axis"};
  paddle::SmallVector<const char*> outputs = {"Out"};

  const char* kernel_name;
  if (ctx.IsDenseTensorVectorInput("Input")) {
    kernel_name = "strided_slice_array";
  } else {
    kernel_name = "strided_slice_raw";
  }
  // NOTE(dev): Use this to avoid regularization.
  KernelSignature sig(kernel_name, inputs, attrs, outputs);
  return sig;
}

KernelSignature StridedSliceGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  const auto& starts = paddle::any_cast<std::vector<int>>(ctx.Attr("starts"));
  const auto& ends = paddle::any_cast<std::vector<int>>(ctx.Attr("ends"));
  const auto& strides = paddle::any_cast<std::vector<int>>(ctx.Attr("strides"));

  bool use_attr_starts = !ctx.IsRuntime() && !starts.empty();
  bool use_attr_ends = !ctx.IsRuntime() && !ends.empty();
  bool use_attr_strides = !ctx.IsRuntime() && !strides.empty();

  const char* starts_key =
      ctx.HasInput("StartsTensor")
          ? "StartsTensor"
          : (ctx.InputSize("StartsTensorList") > 0
                 ? (use_attr_starts ? "starts" : "StartsTensorList")
                 : "starts");
  const char* ends_key =
      ctx.HasInput("EndsTensor")
          ? "EndsTensor"
          : (ctx.InputSize("EndsTensorList") > 0
                 ? (use_attr_ends ? "ends" : "EndsTensorList")
                 : "ends");
  const char* strides_key =
      ctx.HasInput("StridesTensor")
          ? "StridesTensor"
          : (ctx.InputSize("StridesTensorList") > 0
                 ? (use_attr_strides ? "strides" : "StridesTensorList")
                 : "strides");

  paddle::SmallVector<const char*> inputs = {"Input", "Out@GRAD"};
  paddle::SmallVector<const char*> attrs = {"axes",
                                            starts_key,
                                            ends_key,
                                            strides_key,
                                            "infer_flags",
                                            "decrease_axis"};
  paddle::SmallVector<const char*> outputs = {"Input@GRAD"};

  const char* kernel_name;
  if (ctx.IsDenseTensorVectorInput("Input")) {
    kernel_name = "strided_slice_array_grad";
  } else {
    kernel_name = "strided_slice_raw_grad";
  }

  // NOTE(dev): Use this to avoid regularization.
  KernelSignature sig(kernel_name, inputs, attrs, outputs);
  return sig;
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(strided_slice, phi::StridedSliceOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(strided_slice_grad,
                           phi::StridedSliceGradOpArgumentMapping);

/*
******************************************************************
NOTE: The following codes are for 'get_compat_kernel_signature.py'
      DO NOT EDIT IT if you don't know the mechanism.
******************************************************************

############################  Forward ############################

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensor",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensor",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "ends", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensor", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"starts","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "ends",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "StartsTensorList", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensor", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensor", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensorList", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "EndsTensorList", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "ends", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_raw", {"Input"},
              {"axes", "starts", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensor",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensor",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "EndsTensorList", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "ends", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensor", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"StartsTensor","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "EndsTensorList",
"starts","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "ends",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "StartsTensorList", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensor", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensor", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensor", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensorList", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensorList",
"StartsTensorList","infer_flags", "decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "EndsTensorList", "starts","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "ends", "StartsTensor","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "ends", "StartsTensorList","infer_flags",
"decrease_axis"},
              {"Out"});

return KernelSignature("strided_slice_array", {"Input"},
              {"axes", "starts", "ends", "starts","infer_flags",
"decrease_axis"},
              {"Out"});
*/
