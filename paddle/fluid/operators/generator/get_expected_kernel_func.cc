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
            platform::is_npu_place(ctx.GetPlace()) ||
            platform::is_xpu_place(ctx.GetPlace()) ||
            platform::is_custom_place(ctx.GetPlace()),
        true,
        platform::errors::InvalidArgument(
            "float16 can only be used on GPU or NPU or MLU or XPU place"));
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

}  // namespace operators
}  // namespace paddle
