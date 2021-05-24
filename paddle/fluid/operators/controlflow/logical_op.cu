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

#include "paddle/fluid/operators/controlflow/logical_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define LOGICAL_BINARY_FUNCTOR(op_name, op)           \
  template <typename T>                               \
  struct op_name##Functor {                           \
    using ELEMENT_TYPE = T;                           \
    HOSTDEVICE bool operator()(const T* args) const { \
      return args[0] op args[1];                      \
    }                                                 \
  };

LOGICAL_BINARY_FUNCTOR(CudaOr, ||)
LOGICAL_BINARY_FUNCTOR(CudaAnd, &&)
LOGICAL_BINARY_FUNCTOR(CudaXor, ^)
#undef LOGICAL_BINARY_FUNCTOR

template <typename T>
struct CudaNotFunctor {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T* args) const { return !args[0]; }
};

template <typename DeviceContext, typename Functor>
class LogicalOpCudaKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using InT = typename Functor::ELEMENT_TYPE;
  using OutT = bool;
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* z = ctx.Output<framework::Tensor>("Out");
    auto* y = ctx.Input<framework::Tensor>("Y");
    z->mutable_data<OutT>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");
    auto functor = Functor();
    std::vector<framework::Tensor*> outs = {z};
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    if (y != nullptr) {
      std::vector<const framework::Tensor*> ins = {x};
      LaunchElementwiseCudaKernel<ElementwiseType::kUnary, InT, OutT>(
          cuda_ctx, ins, &outs, axis, functor);
    } else {
      axis = axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis;
      std::vector<const framework::Tensor*> ins = {x, y};
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, InT, OutT>(
          cuda_ctx, ins, &outs, axis, functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_LOGICAL_CUDA_KERNEL(op_name, func)                         \
  REGISTER_OP_CUDA_KERNEL(op_name,                                          \
                          ops::LogicalOpCudaKernel<plat::CUDADeviceContext, \
                                                   ops::func##Functor<bool>>);

REGISTER_LOGICAL_CUDA_KERNEL(logical_or, CudaOr);
REGISTER_LOGICAL_CUDA_KERNEL(logical_and, CudaAnd);
REGISTER_LOGICAL_CUDA_KERNEL(logical_xor, CudaXor);
REGISTER_LOGICAL_CUDA_KERNEL(logical_not, CudaNot);
#undef REGISTER_LOGICAL_CUDA_KERNEL
