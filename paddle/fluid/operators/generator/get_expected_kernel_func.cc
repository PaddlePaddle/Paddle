/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/operators/generator/get_expected_kernel_func.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/tensor_util.h"
namespace paddle {
namespace operators {

// oneDNN's reduction kernel is optimized only for reducing throughout the
// most outer dims, so in case of another type of reduction, it would be
// better to fallback to native implementation
static bool ReduceOpHasOptimizedOneDNNKernel(
    const framework::ExecutionContext& ctx) {
  // native reduce kernels don't support bf16
  // so oneDNN kernel is enforced in that case
  if (ctx.Input<phi::DenseTensor>("X")->dtype() == phi::DataType::BFLOAT16)
    return true;

  if (!ctx.HasAttr("dim") || !ctx.HasAttr("reduce_all")) {
    return false;
  }

  auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
  const bool reduce_all = ctx.Attr<bool>("reduce_all");
  int ndims = ctx.Input<phi::DenseTensor>("X")->dims().size();

  if (reduce_all) {
    return true;
  }

  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    if (reduce_dims[i] < 0) reduce_dims[i] = ndims + reduce_dims[i];
  }
  sort(reduce_dims.begin(), reduce_dims.end());
  for (size_t i = 0; i < reduce_dims.size(); ++i) {
    if (reduce_dims[reduce_dims.size() - i - 1] !=
        static_cast<int>(ndims - i - 1)) {
      return false;
    }
  }

  return true;
}

phi::KernelKey GetReduceExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "X");

  if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5 ||
      !ReduceOpHasOptimizedOneDNNKernel(ctx)) {
    op_ptr->SetDnnFallback(true);
  }

  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()) ||
            platform::is_xpu_place(ctx.GetPlace()) ||
            platform::is_custom_place(ctx.GetPlace()),
        true,
        platform::errors::InvalidArgument(
            "float16 can only be used on GPU or NPU or XPU place"));
  }
  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetReduceGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  int out_dtype = ctx.Attr<int>("out_dtype");
  auto input_data_type =
      (out_dtype >= 0)
          ? static_cast<framework::proto::VarType::Type>(out_dtype)
          : op_ptr->IndicateVarDataType(ctx, framework::GradVarName("Out"));
  if (ctx.Input<phi::DenseTensor>("X")->dims().size() > 5) {
    op_ptr->SetDnnFallback(true);
  }

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

phi::KernelKey GetAssignExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  const framework::Variable* var = ctx.InputVar("X");
  if (var->IsType<framework::LoDTensorArray>()) {
    auto t_arr = var->Get<framework::LoDTensorArray>();
    // NOTE(liym27): Support an empty tensor array as Input.
    // And set the kernel type is float.
    if (t_arr.size() == 0) {
      return phi::KernelKey(framework::proto::VarType::FP32,
                            ctx.device_context().GetPlace());
    }
  }
  return phi::KernelKey(
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
      ctx.device_context().GetPlace());
}

phi::KernelKey GetSgdExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto data_type = op_ptr->IndicateVarDataType(ctx, "Param");

  // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_MKLDNN
  const auto* param_var = ctx.InputVar("Param");
  const auto* grad_var = ctx.InputVar("Grad");

  // supported cases
  bool dense_param_sparse_grad = param_var->IsType<phi::DenseTensor>() &&
                                 grad_var->IsType<phi::SelectedRows>();
  bool dense_param_and_grad = param_var->IsType<phi::DenseTensor>() &&
                              grad_var->IsType<phi::DenseTensor>();
  if (!(dense_param_sparse_grad || dense_param_and_grad)) {
    op_ptr->SetDnnFallback(true);
  }
  // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_MKLDNN

  return phi::KernelKey(data_type, ctx.GetPlace());
}

phi::KernelKey GetSoftmaxExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  std::string data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
  auto input_data_type = op_ptr->IndicateVarDataType(ctx, "X");
  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()) ||
            platform::is_xpu_place(ctx.GetPlace()) ||
            platform::is_custom_place(ctx.GetPlace()),
        true,
        platform::errors::InvalidArgument(
            "float16 can only be used on GPU/XPU and custom place"));
  }
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(input_data_type));
}

phi::KernelKey GetSoftmaxGradExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  // choose cudnn kernel if the runtime supported.
  std::string data_format = ctx.Attr<std::string>("data_format");
  phi::DataLayout layout_ = phi::StringToDataLayout(data_format);
  auto input_data_type =
      op_ptr->IndicateVarDataType(ctx, framework::GradVarName("Out"));
  if (input_data_type == framework::proto::VarType::FP16) {
    if (!(platform::is_gpu_place(ctx.GetPlace()) ||
          platform::is_xpu_place(ctx.GetPlace()) ||
          platform::is_custom_place(ctx.GetPlace())))
      PADDLE_THROW(platform::errors::InvalidArgument(
          "float16 can only be used on GPU/XPU and custom place"));
  }
  return phi::KernelKey(
      ctx.GetPlace(), layout_, phi::TransToPhiDataType(input_data_type));
}

phi::KernelKey GetUpdateLossScalingExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto dtype = framework::proto::VarType::FP32;
  if (ctx.MultiInputVar("X").size() >= 1) {
    dtype = op_ptr->IndicateVarDataType(ctx, "X");
  }
  return phi::KernelKey(dtype, ctx.GetPlace());
}

phi::KernelKey GetMatrixNmsExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "Scores"),
                        platform::CPUPlace());
}

phi::KernelKey GetYoloLossExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  return phi::KernelKey(op_ptr->IndicateVarDataType(ctx, "X"),
                        platform::CPUPlace());
}

phi::KernelKey GetUniqueExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  (void)ctx;
  // Return CPUPlace when Attr("is_sorted") is false. Because it means
  // that fluid.layers.unique is called, but there is no cuda kernel.
  if (!ctx.Attr<bool>("is_sorted")) {
    return phi::KernelKey(
        op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  } else {
    // new version paddle.unique is called.
    return phi::KernelKey(
        op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.GetPlace());
  }
}

phi::KernelKey GetAddNExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto x_vars = ctx.MultiInputVar("X");
  auto x_vars_name = ctx.InputNames("X");

  PADDLE_ENFORCE_GT(
      x_vars.size(),
      0,
      platform::errors::InvalidArgument("Input[X] should not be empty"));

  PADDLE_ENFORCE_NOT_NULL(
      x_vars[0],
      platform::errors::NotFound("Input var[%s] should not be nullptr",
                                 x_vars_name[0]));

  if (x_vars[0]->IsType<phi::DenseTensor>()) {
    int dtype = -1;
    for (size_t idx = 0; idx < x_vars.size(); ++idx) {
      PADDLE_ENFORCE_NOT_NULL(
          x_vars[idx],
          platform::errors::NotFound("Input var[%s] should not be nullptr",
                                     x_vars_name[idx]));
      auto tensor =
          framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_vars[idx]);
      if (!tensor->IsInitialized()) {
        continue;
      }
      if (dtype == -1) {
        dtype = framework::TransToProtoVarType(tensor->dtype());
      } else {
        PADDLE_ENFORCE_EQ(dtype,
                          framework::TransToProtoVarType(tensor->dtype()),
                          platform::errors::InvalidArgument(
                              "The inputs type of sum op must be same"));
      }
    }
    PADDLE_ENFORCE_NE(dtype,
                      -1,
                      platform::errors::InvalidArgument(
                          "Sum operator should have at least one tensor"));

    auto data_type = static_cast<framework::proto::VarType::Type>(dtype);

    // NOTE(jiahongyu): Below codes originally enclosed by PADDLE_WITH_MKLDNN
    if (!((data_type == framework::proto::VarType::FP32 ||
           data_type == framework::proto::VarType::BF16) &&
          ctx.OutputVar("Out")->IsType<phi::DenseTensor>())) {
      op_ptr->SetDnnFallback(true);
    } else if (!std::all_of(x_vars.begin(),
                            x_vars.end(),
                            [](const framework::Variable* v) {
                              return v->IsType<phi::DenseTensor>();
                            })) {
      op_ptr->SetDnnFallback(true);
    }
    // NOTE(jiahongyu): Above codes originally enclosed by PADDLE_WITH_MKLDNN

    return phi::KernelKey(data_type, ctx.GetPlace());
  } else if (x_vars[0]->IsType<phi::SelectedRows>()) {
    for (auto& var : x_vars) {
      auto& value = var->Get<phi::SelectedRows>().value();
      if (value.IsInitialized()) {
        return phi::KernelKey(framework::TransToProtoVarType(value.dtype()),
                              ctx.GetPlace());
      }
    }
    // if input sparse vars are not initialized, use an default kernel type.
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  } else if (x_vars[0]->IsType<framework::LoDTensorArray>()) {
    for (auto& x_var : x_vars) {
      auto& array = x_var->Get<framework::LoDTensorArray>();
      for (auto& each : array) {
        if (each.numel() != 0 && each.IsInitialized()) {
          return phi::KernelKey(framework::TransToProtoVarType(each.dtype()),
                                ctx.GetPlace());
        }
      }
    }
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Expected each tensor in Input(x) in sum op has be initialized, but "
        "some tensor in Input(x) is not be initialized, please check your "
        "code.",
        framework::ToTypeName(x_vars[0]->Type())));
  }
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Expected type of Input(X) must be Tensor,  SelectedRows or "
      "LodTensorArray. But got "
      "unsupport type: %s.",
      framework::ToTypeName(x_vars[0]->Type())));
}

phi::KernelKey GetInstanceNormExpectedKernelType(
    const framework::ExecutionContext& ctx,
    const framework::OperatorWithKernel* op_ptr) {
  auto input_data_type =
      op_ptr->OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  if (ctx.HasInput("Scale")) {
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Scale")->dtype()),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
  }
  if (ctx.HasInput("Bias")) {
    PADDLE_ENFORCE_EQ(in_param_type,
                      framework::TransToProtoVarType(
                          ctx.Input<phi::DenseTensor>("Bias")->dtype()),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
  }

  return phi::KernelKey(input_data_type, ctx.GetPlace());
}

}  // namespace operators
}  // namespace paddle
