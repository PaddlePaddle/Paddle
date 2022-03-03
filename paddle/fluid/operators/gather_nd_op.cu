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
#include "paddle/fluid/operators/gather_nd_op.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"

namespace paddle {
namespace operators {

template <typename T>
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
    const auto &index_type = index->dtype();
    bool index_type_match = index_type == phi::DataType::INT32 ||
                            index_type == phi::DataType::INT64;
    PADDLE_ENFORCE_EQ(
        index_type_match, true,
        platform::errors::InvalidArgument(
            "Index holds the wrong type, it holds [%s], but "
            "desires to be [%s] or [%s].",
            index_type, phi::DataType::INT32, phi::DataType::INT64));
    auto &dev_ctx = ctx.cuda_device_context();
    if (index_type == phi::DataType::INT32) {
      phi::funcs::GPUGatherNd<T, int>(dev_ctx, *x, *index, output);
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::GPUGatherNd<T, int64_t>(dev_ctx, *x, *index, output);
    }
  }
};

template <typename T>
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

    const auto &index_type = index->dtype();
    bool index_type_match = index_type == phi::DataType::INT32 ||
                            index_type == phi::DataType::INT64;

    PADDLE_ENFORCE_EQ(
        index_type_match, true,
        platform::errors::InvalidArgument(
            "Index holds the wrong type, it holds [%s],"
            "but desires to be [%s] or [%s].",
            index_type, phi::DataType::INT32, phi::DataType::INT64));

    auto &dev_ctx = ctx.cuda_device_context();
    if (index_type == phi::DataType::INT32) {
      phi::funcs::GPUScatterNdAdd<T, int>(dev_ctx, *dO, *index, dX);
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::GPUScatterNdAdd<T, int64_t>(dev_ctx, *dO, *index, dX);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(gather_nd, ops::GatherNdOpCUDAKernel<float>,
                        ops::GatherNdOpCUDAKernel<double>,
                        ops::GatherNdOpCUDAKernel<int64_t>,
                        ops::GatherNdOpCUDAKernel<int>,
                        ops::GatherNdOpCUDAKernel<int16_t>,
                        ops::GatherNdOpCUDAKernel<bool>,
                        ops::GatherNdOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(gather_nd_grad, ops::GatherNdGradOpCUDAKernel<float>,
                        ops::GatherNdGradOpCUDAKernel<double>,
                        ops::GatherNdGradOpCUDAKernel<int64_t>,
                        ops::GatherNdGradOpCUDAKernel<int>,
                        ops::GatherNdGradOpCUDAKernel<plat::float16>);
