/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/operators/scatter.cu.h"
#include "paddle/fluid/operators/scatter_nd_add_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ScatterNdAddOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *X = ctx.Input<Tensor>("X");
    auto *Ids = ctx.Input<Tensor>("Index");
    auto *Updates = ctx.Input<Tensor>("Updates");
    auto *Out = ctx.Output<Tensor>("Out");

    framework::TensorCopySync(*X, ctx.GetPlace(), Out);
    const auto &index_type = framework::TransToProtoVarType(Ids->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s], but "
                          "desires to be [%s] or [%s].",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    if (index_type == framework::proto::VarType::INT32) {
      GPUScatterNdAdd<DeviceContext, T, int32_t>(ctx, *Updates, *Ids, Out);
    } else {
      GPUScatterNdAdd<DeviceContext, T, int64_t>(ctx, *Updates, *Ids, Out);
    }
  }
};

template <typename DeviceContext, typename T>
class ScatterNdAddGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dUpdates = ctx.Output<Tensor>(framework::GradVarName("Updates"));
    auto *Ids = ctx.Input<Tensor>("Index");
    auto *dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    if (dX) {
      framework::TensorCopy(*dOut, ctx.GetPlace(), dX);
    }
    if (dUpdates) {
      dUpdates->mutable_data<T>(ctx.GetPlace());
      // Gradient by Gather
      const auto &index_type = framework::TransToProtoVarType(Ids->dtype());
      if (index_type == framework::proto::VarType::INT32) {
        GPUGatherNd<DeviceContext, T, int32_t>(ctx, *dOut, *Ids, dUpdates);
      } else {
        GPUGatherNd<DeviceContext, T, int64_t>(ctx, *dOut, *Ids, dUpdates);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(scatter_nd_add,
                        ops::ScatterNdAddOpCUDAKernel<CUDA, float>,
                        ops::ScatterNdAddOpCUDAKernel<CUDA, double>,
                        ops::ScatterNdAddOpCUDAKernel<CUDA, int64_t>,
                        ops::ScatterNdAddOpCUDAKernel<CUDA, int>,
                        ops::ScatterNdAddOpCUDAKernel<CUDA, plat::float16>);

REGISTER_OP_CUDA_KERNEL(scatter_nd_add_grad,
                        ops::ScatterNdAddGradOpCUDAKernel<CUDA, float>,
                        ops::ScatterNdAddGradOpCUDAKernel<CUDA, double>,
                        ops::ScatterNdAddGradOpCUDAKernel<CUDA, int64_t>,
                        ops::ScatterNdAddGradOpCUDAKernel<CUDA, int>,
                        ops::ScatterNdAddGradOpCUDAKernel<CUDA, plat::float16>);
