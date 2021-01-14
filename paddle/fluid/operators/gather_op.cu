/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/operators/scatter.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class GatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Out");

    if (ctx.HasInput("Axis")) {
      const Tensor *axis = ctx.Input<Tensor>("Axis");
      const auto &index_type = index->type();
      const auto &axis_type = axis->type();
      auto place = ctx.GetPlace();
      if (index_type == framework::proto::VarType::INT32 &&
          axis_type == framework::proto::VarType::INT32) {
        GatherV2CUDAFunction<T, int32_t, int32_t>(x, index, axis, output, place,
                                                  ctx);
      }
      if (index_type == framework::proto::VarType::INT32 &&
          axis_type == framework::proto::VarType::INT64) {
        GatherV2CUDAFunction<T, int32_t, int64_t>(x, index, axis, output, place,
                                                  ctx);
      }
      if (index_type == framework::proto::VarType::INT64 &&
          axis_type == framework::proto::VarType::INT32) {
        GatherV2CUDAFunction<T, int64_t, int32_t>(x, index, axis, output, place,
                                                  ctx);
      }
      if (index_type == framework::proto::VarType::INT64 &&
          axis_type == framework::proto::VarType::INT64) {
        GatherV2CUDAFunction<T, int64_t, int64_t>(x, index, axis, output, place,
                                                  ctx);
      }
      return;
    }
    output->mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;
    const auto &index_type = index->type();
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
      GPUGather<T, int>(ctx.device_context(), *x, *index, output);
    } else if (index_type == framework::proto::VarType::INT64) {
      GPUGather<T, int64_t>(ctx.device_context(), *x, *index, output);
    }
  }
};

template <typename T>
class GatherGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));
    auto *index = ctx.Input<Tensor>("Index");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));

    if (ctx.HasInput("Axis")) {
      const Tensor *axis = ctx.Input<Tensor>("Axis");
      const auto &index_type = index->type();
      const auto &axis_type = axis->type();
      auto place = ctx.GetPlace();
      if (index_type == framework::proto::VarType::INT32 &&
          axis_type == framework::proto::VarType::INT32) {
        GatherV2GradCUDAFunction<T, int32_t, int32_t>(dO, index, axis, dX,
                                                      place, ctx);
      }
      if (index_type == framework::proto::VarType::INT32 &&
          axis_type == framework::proto::VarType::INT64) {
        GatherV2GradCUDAFunction<T, int32_t, int64_t>(dO, index, axis, dX,
                                                      place, ctx);
      }
      if (index_type == framework::proto::VarType::INT64 &&
          axis_type == framework::proto::VarType::INT32) {
        GatherV2GradCUDAFunction<T, int64_t, int32_t>(dO, index, axis, dX,
                                                      place, ctx);
      }
      if (index_type == framework::proto::VarType::INT64 &&
          axis_type == framework::proto::VarType::INT64) {
        GatherV2GradCUDAFunction<T, int64_t, int64_t>(dO, index, axis, dX,
                                                      place, ctx);
      }
      return;
    }

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;

    const auto &index_type = index->type();
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
      GPUScatterAssign<T, int>(ctx, *dO, *index, dX,
                               ctx.Attr<bool>("overwrite"));
    } else if (index_type == framework::proto::VarType::INT64) {
      GPUScatterAssign<T, int64_t>(ctx, *dO, *index, dX,
                                   ctx.Attr<bool>("overwrite"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(gather, ops::GatherOpCUDAKernel<float>,
                        ops::GatherOpCUDAKernel<double>,
                        ops::GatherOpCUDAKernel<int64_t>,
                        ops::GatherOpCUDAKernel<int>,
                        ops::GatherOpCUDAKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(gather_grad, ops::GatherGradOpCUDAKernel<float>,
                        ops::GatherGradOpCUDAKernel<double>,
                        ops::GatherGradOpCUDAKernel<int64_t>,
                        ops::GatherGradOpCUDAKernel<int>,
                        ops::GatherGradOpCUDAKernel<plat::float16>);
