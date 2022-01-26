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

#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename Functor, typename InverseFunctor>
class CompareOpKernel<platform::CUDADeviceContext, Functor, InverseFunctor>
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  using InT = typename Functor::ELEM_TYPE;
  using OutT = bool;
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto functor = Functor();
    std::vector<const framework::Tensor*> ins;
    std::vector<framework::Tensor*> outs;
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    int axis = PackTensorsIntoVector<OutT>(ctx, &ins, &outs);
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                   InT, OutT>(
        cuda_ctx, ins, &outs, axis, functor);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_CUDA_COMPARE_KERNEL(op_type, func)                            \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      op_type,                                                                 \
      ops::CompareOpKernel<plat::CUDADeviceContext, ops::func<bool>, void>,    \
      ops::CompareOpKernel<plat::CUDADeviceContext, ops::func<int>, void>,     \
      ops::CompareOpKernel<plat::CUDADeviceContext, ops::func<int64_t>, void>, \
      ops::CompareOpKernel<plat::CUDADeviceContext, ops::func<float>, void>,   \
      ops::CompareOpKernel<plat::CUDADeviceContext, ops::func<double>, void>);

REGISTER_CUDA_COMPARE_KERNEL(equal, EqualFunctor)
REGISTER_CUDA_COMPARE_KERNEL(not_equal, NotEqualFunctor)
REGISTER_CUDA_COMPARE_KERNEL(less_than, LessThanFunctor)
REGISTER_CUDA_COMPARE_KERNEL(less_equal, LessEqualFunctor)
REGISTER_CUDA_COMPARE_KERNEL(greater_than, GreaterThanFunctor)
REGISTER_CUDA_COMPARE_KERNEL(greater_equal, GreaterEqualFunctor)
#undef REGISTER_CUDA_COMPARE_KERNEL
