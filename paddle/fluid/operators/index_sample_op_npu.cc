/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {
<<<<<<< HEAD

template <typename IndexT>
void IndexSampleGather(const paddle::platform::NPUDeviceContext& dev_ctx,
                       const phi::DenseTensor* index,
                       const phi::DenseTensor* input,
                       phi::DenseTensor* out) {
=======
using Tensor = framework::Tensor;

template <typename IndexT>
void IndexSampleGather(const paddle::platform::NPUDeviceContext& dev_ctx,
                       const Tensor* index,
                       const Tensor* input,
                       Tensor* out) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto index_dims = index->dims();
  auto input_dims = input->dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];

  std::vector<IndexT> gather_index_vec;
  std::vector<IndexT> index_vec;
  framework::TensorToVector(*index, dev_ctx, &index_vec);
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      gather_index_vec.push_back(i);
      gather_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }
<<<<<<< HEAD
  phi::DenseTensor gather_index;
=======
  Tensor gather_index;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  framework::TensorFromVector(gather_index_vec, dev_ctx, &gather_index);
  gather_index.Resize({batch_size, index_length, 2});

  NpuOpRunner runner;
  runner.SetType("GatherNd")
      .AddInput(*input)
      .AddInput(gather_index)
      .AddOutput(*out);
  runner.Run(dev_ctx.stream());
}

template <typename T>
class IndexSampleNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
<<<<<<< HEAD
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* index = ctx.Input<phi::DenseTensor>("Index");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
=======
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* index = ctx.Input<framework::LoDTensor>("Index");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    out->mutable_data<T>(ctx.GetPlace());

    const auto& index_type = framework::TransToProtoVarType(index->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleGather<int32_t>(dev_ctx, index, input, out);
    } else {
      IndexSampleGather<int64_t>(dev_ctx, index, input, out);
    }
  }
};

template <typename IndexT>
void IndexSampleGradScatter(const paddle::platform::NPUDeviceContext& dev_ctx,
<<<<<<< HEAD
                            const phi::DenseTensor* index,
                            const phi::DenseTensor* out_grad,
                            phi::DenseTensor* x_grad) {
=======
                            const Tensor* index,
                            const Tensor* out_grad,
                            Tensor* x_grad) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  auto index_dims = index->dims();
  auto input_dims = x_grad->dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];

  std::vector<IndexT> scatter_index_vec;
  std::vector<IndexT> index_vec;
  framework::TensorToVector(*index, dev_ctx, &index_vec);
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      scatter_index_vec.push_back(i);
      scatter_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }
<<<<<<< HEAD
  phi::DenseTensor scatter_index;
=======
  Tensor scatter_index;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  framework::TensorFromVector(scatter_index_vec, dev_ctx, &scatter_index);
  scatter_index.Resize({batch_size, index_length, 2});

  NpuOpRunner runner;
  runner.SetType("ScatterNd")
      .AddInput(scatter_index)
      .AddInput(*out_grad)
      .AddInput(phi::vectorize<IndexT>(x_grad->dims()))
      .AddOutput(*x_grad);
  runner.Run(dev_ctx.stream());
}

template <typename T>
class IndexSampleGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
<<<<<<< HEAD
    auto* index = ctx.Input<phi::DenseTensor>("Index");
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
=======
    auto* index = ctx.Input<framework::LoDTensor>("Index");
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    x_grad->mutable_data<T>(ctx.GetPlace());

    const auto& index_type = framework::TransToProtoVarType(index->dtype());
    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleGradScatter<int32_t>(dev_ctx, index, out_grad, x_grad);
    } else {
      IndexSampleGradScatter<int64_t>(dev_ctx, index, out_grad, x_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(index_sample,
                       ops::IndexSampleNPUKernel<plat::float16>,
                       ops::IndexSampleNPUKernel<float>,
                       ops::IndexSampleNPUKernel<int32_t>,
                       ops::IndexSampleNPUKernel<int64_t>);
REGISTER_OP_NPU_KERNEL(index_sample_grad,
                       ops::IndexSampleGradNPUKernel<plat::float16>,
                       ops::IndexSampleGradNPUKernel<float>,
                       ops::IndexSampleGradNPUKernel<int32_t>,
                       ops::IndexSampleGradNPUKernel<int64_t>);
