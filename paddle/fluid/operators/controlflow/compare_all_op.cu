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

#include <thrust/fill.h>
#include "paddle/fluid/operators/controlflow/compare_all_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"

namespace paddle {
namespace operators {

template <typename T>
struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  inline T initial() { return static_cast<T>(true); }

  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a & b;
  }
};

template <typename DeviceContext, typename Functor>
class CompareReduceOpKernel
    : public framework::OpKernel<typename Functor::ELEM_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using T = typename Functor::ELEM_TYPE;
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* z = context.Output<Tensor>("Out");
    bool* z_data = z->mutable_data<bool>(context.GetPlace());
    Tensor tmp;

    if (x->dims() != y->dims()) {
      thrust::device_ptr<bool> z_dev_ptr(z_data);
      thrust::fill(z_dev_ptr, z_dev_ptr + 1, false);
      return;
    } else {
      tmp.mutable_data<bool>(x->dims(), context.GetPlace());
      const auto& cuda_ctx =
          context.template device_context<platform::CUDADeviceContext>();
      std::vector<const framework::Tensor*> ins = {x, y};
      std::vector<framework::Tensor*> outs = {&tmp};
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<bool>(
          cuda_ctx, ins, &outs, Functor());

      // Reduce by 'bitwise and' operator
      std::vector<int> reduce_dims;
      reduce_dims.resize(tmp.dims().size());
      for (int i = 0; i < reduce_dims.size(); ++i) reduce_dims[i] = i;
      auto stream = context.cuda_device_context().stream();
      TensorReduceImpl<bool, bool, BitwiseAdd, kps::IdentityFunctor<bool>>(
          context.cuda_device_context(), tmp, z, kps::IdentityFunctor<bool>(),
          reduce_dims, stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_COMPARE_REDUCE_CUDA_KERNEL(op_type, functor)                  \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      op_type,                                                                 \
      ops::CompareReduceOpKernel<plat::CUDADeviceContext, ops::functor<bool>>, \
      ops::CompareReduceOpKernel<plat::CUDADeviceContext, ops::functor<int>>,  \
      ops::CompareReduceOpKernel<plat::CUDADeviceContext,                      \
                                 ops::functor<int64_t>>,                       \
      ops::CompareReduceOpKernel<plat::CUDADeviceContext,                      \
                                 ops::functor<float>>,                         \
      ops::CompareReduceOpKernel<plat::CUDADeviceContext,                      \
                                 ops::functor<double>>);

REGISTER_COMPARE_REDUCE_CUDA_KERNEL(equal_all, EqualReduceFunctor)
#undef REGISTER_COMPARE_REDUCE_CUDA_KERNEL
