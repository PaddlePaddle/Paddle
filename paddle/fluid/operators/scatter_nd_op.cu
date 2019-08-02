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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/scatter_nd.cu.h"
#include "paddle/fluid/operators/scatter_nd_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ScatterNDOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *X = ctx.Input<Tensor>("X");
    auto *Ids = ctx.Input<Tensor>("Ids");
    auto *Updates = ctx.Input<Tensor>("Updates");
    auto *Out = ctx.Output<Tensor>("Out");
    int dim = ctx.Attr<int>("dim");

    Out->ShareDataWith(*X);
    GPUScatterNDAssign<DeviceContext, T>(ctx, *X, *Updates, *Ids, Out, dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(scatter_nd, ops::ScatterNDOpCUDAKernel<CUDA, float>,
                        ops::ScatterNDOpCUDAKernel<CUDA, double>,
                        ops::ScatterNDOpCUDAKernel<CUDA, int>);
