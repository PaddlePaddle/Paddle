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
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using DataLayout = platform::DataLayout;
using helper = phi::CudnnFrontendConvHelper;
using TensorDesc_t = cudnn_frontend::Tensor;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class FusedScaleBiasAddReluOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
        phi::autotune::AlgorithmType::kScaleBiasAddRelu);

    // attributes
    bool fuse_dual = ctx.Attr<bool>("fuse_dual");
    // required input variables
    auto* x1_tensor = ctx.Input<Tensor>("X1");
    auto* scale1_tensor = ctx.Input<Tensor>("Scale1");
    auto* bias1_tensor = ctx.Input<Tensor>("Bias1");
    auto* x2_tensor = ctx.Input<Tensor>("X2");
    // dispensable inputs
    const Tensor* scale2_tensor = nullptr;
    const Tensor* bias2_tensor = nullptr;
    if (fuse_dual) {
      scale2_tensor = ctx.Input<Tensor>("Scale2");
      bias2_tensor = ctx.Input<Tensor>("Bias2");
    }
    // outputs
    auto* y_tensor = ctx.Output<Tensor>("Y");
    y_tensor->mutable_data<T>(ctx.GetPlace());
    // get handles
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    // create tensor descriptors
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format =
        phi::backends::gpu::ToCudnnDataType(x1_tensor->dtype());
    auto tensor_format_math = CUDNN_DATA_FLOAT;
    auto compute_dtype = CUDNN_DATA_FLOAT;

    auto dim_x = phi::backends::gpu::TransformDimOrder(
        phi::vectorize<int64_t>(x1_tensor->dims()));
    std::vector<int64_t> dim_c(dim_x.size(), 1);
    dim_c[1] = dim_x[1];  //  [1, C, 1, 1]

    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    int64_t uid = 100;

    // inputs
    auto x1_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(x1_tensor->data<T>()));
    uids.push_back(uid++);

    auto x2_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(x2_tensor->data<T>()));
    uids.push_back(uid++);

    auto scale1_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(scale1_tensor->data<T>()));
    uids.push_back(uid++);

    auto bias1_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(bias1_tensor->data<T>()));
    uids.push_back(uid++);

    // dispensable inputs
    auto scale2_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<T*>(scale2_tensor->data<T>()));
      uids.push_back(uid);
    }
    uid++;

    auto bias2_desc = helper::GetGeneralTensorDescriptor(
        dim_c, layout_format, uid, 16, tensor_format);
    if (fuse_dual) {
      data_ptrs.push_back(const_cast<T*>(bias2_tensor->data<T>()));
      uids.push_back(uid);
    }
    uid++;

    // outputs
    auto y_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(y_tensor->data<T>());
    uids.push_back(uid++);

    // virtual outputs
    auto after_scale1 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_bias1 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_scale2 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_bias2 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_add = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    // build ops
    auto scale1_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_MUL, compute_dtype, x1_desc, scale1_desc, after_scale1);

    auto bias1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                            compute_dtype,
                                            after_scale1,
                                            bias1_desc,
                                            after_bias1);

    auto scale2_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_MUL, compute_dtype, x2_desc, scale2_desc, after_scale2);

    auto bias2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                            compute_dtype,
                                            after_scale2,
                                            bias2_desc,
                                            after_bias2);

    TensorDesc_t* tensor_to_add = fuse_dual ? &after_bias2 : &x2_desc;

    auto add_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                          compute_dtype,
                                          after_bias1,
                                          *tensor_to_add,
                                          after_add);

    auto relu_desc = cudnn_frontend::PointWiseDescBuilder()
                         .setMode(CUDNN_POINTWISE_RELU_FWD)
                         .setComputeType(compute_dtype)
                         .build();

    auto relu_op = cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(after_add)
                       .setyDesc(y_desc)
                       .setpwDesc(relu_desc)
                       .build();

    // build op graph
    std::vector<cudnn_frontend::Operation const*> ops;
    if (fuse_dual) {
      ops = std::vector<cudnn_frontend::Operation const*>(
          {&scale1_op, &bias1_op, &scale2_op, &bias2_op, &add_op, &relu_op});
    } else {
      ops = std::vector<cudnn_frontend::Operation const*>(
          {&scale1_op, &bias1_op, &add_op, &relu_op});
    }

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(ops.size(), ops.data())
                        .build();
    VLOG(6) << op_graph.describe();

    cudnn_frontend::feature_vector_t feature_vector;
    phi::autotune::BuildFeatureVector(&feature_vector, dim_x, fuse_dual);

    if (plan_cache.FindPlan(feature_vector)) {
      const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
      int64_t workspace_size = 0;
      plan_cache.GetPlan(feature_vector, &cached_plan, &workspace_size);
      helper::ExecutePlan(handle,
                          &workspace_handle,
                          &data_ptrs,
                          &uids,
                          cached_plan->get_raw_desc(),
                          workspace_size);
      return;
    }

    auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
    VLOG(6) << "Plan tag: " << plan.getTag();

    auto workspace_size = plan.getWorkspaceSize();
    VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        plan.get_raw_desc(),
                        workspace_size);

    plan_cache.InsertPlan(feature_vector, plan);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_scale_bias_add_relu,
    ops::FusedScaleBiasAddReluOpKernel<paddle::platform::float16>);
