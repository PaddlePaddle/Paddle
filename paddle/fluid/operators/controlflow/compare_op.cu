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

#define DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(Func, op) \
  template <typename T, typename Enable = void>               \
  struct Func##Functor {                                      \
    using ELEMENT_TYPE = T;                                   \
    inline HOSTDEVICE T operator()(const T* args) const {     \
      return args[0] op args[1];                              \
    }                                                         \
  };

DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaLessThan, <)
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaLessEqual, <=)
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaGreaterThan, >)
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaGreaterEqual, >=)
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaEqual, ==)
DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT(CudaNotEqual, !=)
#undef DEFINE_CMP_BINARY_FUNCTOR_WITH_PONTER_INPUT

template <typename T>
struct CudaEqualFunctor<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T* args) const {
    return fabs(static_cast<double>(args[0] - args[1])) < 1e-8;
  }
};

template <typename T>
struct CudaNotEqualFunctor<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T* args) const {
    return fabs(static_cast<double>(args[0] - args[1])) > 1e-8;
  }
};

template <typename DeviceContext, typename Functor>
class CompareOpCudaKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using InT = typename Functor::ELEMENT_TYPE;
  using OutT = bool;
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* z = ctx.Output<framework::Tensor>("Out");
    z->mutable_data<OutT>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");
    axis = axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis;
    auto functor = Functor();

    std::vector<const framework::Tensor*> ins = {x, y};
    std::vector<framework::Tensor*> outs = {z};
    const auto& cuda_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, InT, OutT>(
        cuda_ctx, ins, &outs, axis, functor);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_CUDA_COMPARE_KERNEL(op_type, func)               \
  REGISTER_OP_CUDA_KERNEL(                                        \
      op_type, ops::CompareOpCudaKernel<plat::CUDADeviceContext,  \
                                        ops::func##Functor<int>>, \
      ops::CompareOpCudaKernel<plat::CUDADeviceContext,           \
                               ops::func##Functor<int64_t>>,      \
      ops::CompareOpCudaKernel<plat::CUDADeviceContext,           \
                               ops::func##Functor<float>>,        \
      ops::CompareOpCudaKernel<plat::CUDADeviceContext,           \
                               ops::func##Functor<double>>);

REGISTER_CUDA_COMPARE_KERNEL(equal, CudaEqual)
REGISTER_CUDA_COMPARE_KERNEL(not_equal, CudaNotEqual)
REGISTER_CUDA_COMPARE_KERNEL(less_than, CudaLessThan)
REGISTER_CUDA_COMPARE_KERNEL(less_equal, CudaLessEqual)
REGISTER_CUDA_COMPARE_KERNEL(greater_than, CudaGreaterThan)
REGISTER_CUDA_COMPARE_KERNEL(greater_equal, CudaGreaterEqual)
#undef REGISTER_CUDA_COMPARE_KERNEL
