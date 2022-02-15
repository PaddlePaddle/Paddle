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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/gather_nd_op.h"
#include "paddle/fluid/operators/scatter.cu.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class GatherNdOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Out");

    output->mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;
    const auto &index_type = framework::TransToProtoVarType(index->dtype());
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
      GPUGatherNd<DeviceContext, T, int>(ctx, *x, *index, output);
    } else if (index_type == framework::proto::VarType::INT64) {
      GPUGatherNd<DeviceContext, T, int64_t>(ctx, *x, *index, output);
    }
  }
};

template <typename DeviceContext, typename T>
class GatherNdGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *index = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;

    const auto &index_type = framework::TransToProtoVarType(index->dtype());
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;

    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Index holds the wrong type, it holds [%s],"
                          "but desires to be [%s] or [%s].",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    if (index_type == framework::proto::VarType::INT32) {
      GPUScatterNdAdd<DeviceContext, T, int>(ctx, *dO, *index, dX);
    } else if (index_type == framework::proto::VarType::INT64) {
      GPUScatterNdAdd<DeviceContext, T, int64_t>(ctx, *dO, *index, dX);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(gather_nd, ops::GatherNdOpCUDAKernel<CUDA, float>,
                        ops::GatherNdOpCUDAKernel<CUDA, double>,
                        ops::GatherNdOpCUDAKernel<CUDA, int64_t>,
                        ops::GatherNdOpCUDAKernel<CUDA, int>,
                        ops::GatherNdOpCUDAKernel<CUDA, bool>,
                        ops::GatherNdOpCUDAKernel<CUDA, plat::float16>);

REGISTER_OP_CUDA_KERNEL(gather_nd_grad,
                        ops::GatherNdGradOpCUDAKernel<CUDA, float>,
                        ops::GatherNdGradOpCUDAKernel<CUDA, double>,
                        ops::GatherNdGradOpCUDAKernel<CUDA, int64_t>,
                        ops::GatherNdGradOpCUDAKernel<CUDA, int>,
                        ops::GatherNdGradOpCUDAKernel<CUDA, plat::float16>);
