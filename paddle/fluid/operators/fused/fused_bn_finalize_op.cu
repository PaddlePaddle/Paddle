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

#include <array>
#include <memory>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;
using helper = phi::CudnnFrontendConvHelper;

template <typename T>
class FusedBnFinalizeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::float16;
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    // attributes
    auto accumulation_count = ctx.Attr<int64_t>("accumulation_count");
    auto momentum = ctx.Attr<float>("momentum");
    float exp_decay = 1. - momentum;
    auto epsilon_f = ctx.Attr<float>("epsilon");
    float epsilon = epsilon_f;
    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon,
                       static_cast<float>(CUDNN_BN_MIN_EPSILON + FLT_EPSILON));
    // inputs
    auto* sum_tensor = ctx.Input<Tensor>("Sum");
    auto* sqsum_tensor = ctx.Input<Tensor>("SqSum");
    auto* scale_tensor = ctx.Input<Tensor>("Scale");
    auto* bias_tensor = ctx.Input<Tensor>("Bias");
    auto* input_running_mean_tensor = ctx.Input<Tensor>("inputRunningMean");
    auto* input_running_var_tensor = ctx.Input<Tensor>("inputRunningVar");
    // outputs
    auto* updated_running_mean_tensor =
        ctx.Output<Tensor>("updatedRunningMean");
    auto* updated_running_var_tensor = ctx.Output<Tensor>("updatedRunningVar");
    auto* saved_mean_tensor = ctx.Output<Tensor>("SavedMean");
    auto* saved_inv_var_tensor = ctx.Output<Tensor>("SavedInvVar");
    auto* eq_scale_tensor = ctx.Output<Tensor>("eqScale");
    auto* eq_bias_tensor = ctx.Output<Tensor>("eqBias");
    updated_running_mean_tensor->mutable_data<T>(ctx.GetPlace());
    updated_running_var_tensor->mutable_data<T>(ctx.GetPlace());
    saved_mean_tensor->mutable_data<T>(ctx.GetPlace());
    saved_inv_var_tensor->mutable_data<T>(ctx.GetPlace());
    eq_scale_tensor->mutable_data<U>(ctx.GetPlace());
    eq_bias_tensor->mutable_data<U>(ctx.GetPlace());
    // prepare handles
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    // set dtypes
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format_bn = platform::ToCudnnDataType(
        framework::TransToProtoVarType(sum_tensor->dtype()));
    auto tensor_format = platform::ToCudnnDataType(
        framework::TransToProtoVarType(eq_scale_tensor->dtype()));
    auto compute_dtype = CUDNN_DATA_FLOAT;
    // create tensor descriptors
    auto dim_input = phi::vectorize<int64_t>(sum_tensor->dims());
    std::vector<int64_t> dim_c = {1, dim_input[0], 1, 1};  //  [1, C, 1, 1]
    std::vector<int64_t> dim_scalar = {1, 1, 1, 1};
    std::vector<int64_t> stride_scalar = {1, 1, 1, 1};

    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    int64_t uid = 100;

    // inputs
    auto sum_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(sum_tensor->data<T>()));
    uids.push_back(uid++);

    auto sqsum_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(sqsum_tensor->data<T>()));
    uids.push_back(uid++);

    auto scale_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(scale_tensor->data<T>()));
    uids.push_back(uid++);

    auto bias_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(bias_tensor->data<T>()));
    uids.push_back(uid++);

    auto input_running_mean_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(input_running_mean_tensor->data<T>()));
    uids.push_back(uid++);

    auto input_running_var_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(const_cast<T*>(input_running_var_tensor->data<T>()));
    uids.push_back(uid++);

    // outputs
    auto updated_running_mean_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(updated_running_mean_tensor->data<T>());
    uids.push_back(uid++);

    auto updated_running_var_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(updated_running_var_tensor->data<T>());
    uids.push_back(uid++);

    auto saved_mean_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(saved_mean_tensor->data<T>());
    uids.push_back(uid++);

    auto saved_inv_var_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format_bn);
    data_ptrs.push_back(saved_inv_var_tensor->data<T>());
    uids.push_back(uid++);

    auto eq_scale_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(eq_scale_tensor->data<U>());
    uids.push_back(uid++);

    auto eq_bias_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(eq_bias_tensor->data<U>());
    uids.push_back(uid++);

    // scalar descriptors
    auto epsilon_desc =
        cudnn_frontend::TensorBuilder()
            .setDim(dim_scalar.size(), dim_scalar.data())
            .setStride(stride_scalar.size(), stride_scalar.data())
            .setId(uid)
            .setAlignment(16)
            .setDataType(CUDNN_DATA_FLOAT)
            .setByValue(true)
            .build();
    data_ptrs.push_back(&epsilon);
    uids.push_back(uid++);

    auto exp_decay_desc =
        cudnn_frontend::TensorBuilder()
            .setDim(dim_scalar.size(), dim_scalar.data())
            .setStride(stride_scalar.size(), stride_scalar.data())
            .setId(uid)
            .setAlignment(16)
            .setDataType(CUDNN_DATA_FLOAT)
            .setByValue(true)
            .build();
    data_ptrs.push_back(&exp_decay);
    uids.push_back(uid++);

    auto accum_count_desc =
        cudnn_frontend::TensorBuilder()
            .setDim(dim_scalar.size(), dim_scalar.data())
            .setStride(stride_scalar.size(), stride_scalar.data())
            .setId(uid)
            .setAlignment(16)
            .setDataType(CUDNN_DATA_INT64)
            .setByValue(true)
            .build();
    data_ptrs.push_back(&accumulation_count);
    uids.push_back(uid++);

    //  build ops
    auto finalize_stat_op =
        cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
            .setComputeType(compute_dtype)
            .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
            .setSumDesc(sum_desc)
            .setSqSumDesc(sqsum_desc)
            .setScaleAndBias(scale_desc, bias_desc)
            .setEqScaleAndBias(eq_scale_desc, eq_bias_desc)
            .setPrevRunningMeanAndVar(input_running_mean_desc,
                                      input_running_var_desc)
            .setNextRunningMeanAndVar(updated_running_mean_desc,
                                      updated_running_var_desc)
            .setSavedMeanAndInvVar(saved_mean_desc, saved_inv_var_desc)
            .setEpsilonTensor(epsilon_desc)
            .setAccumCountTensor(accum_count_desc)
            .setExpDecayFactorTensor(exp_decay_desc)
            .build();

    std::array<cudnn_frontend::Operation const*, 1> ops = {&finalize_stat_op};
    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(ops.size(), ops.data())
                        .build();
    VLOG(6) << op_graph.describe();

    auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
    auto workspace_size = plan.getWorkspaceSize();
    VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        plan.get_raw_desc(),
                        workspace_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_bn_finalize, ops::FusedBnFinalizeOpKernel<float>);
