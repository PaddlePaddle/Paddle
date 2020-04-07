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

#include "paddle/fluid/operators/controlflow/compare_reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
namespace paddle {
namespace operators {

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  template <typename T>
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
    int axis = context.Attr<int>("axis");

    Tensor tmp;
    framework::DDim x_dims = x->dims();
    framework::DDim y_dims = y->dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> tmp_dims_array(max_dim);
    GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                           y_dims_array.data(), tmp_dims_array.data(), max_dim,
                           axis);
    tmp.mutable_data<bool>(framework::make_ddim(tmp_dims_array),
                           context.GetPlace());
    ElementwiseComputeEx<Functor, DeviceContext, T, bool>(context, x, y, axis,
                                                          Functor(), &tmp);
    // Reduce by 'bitwise and' operator
    std::vector<int> reduce_dims;
    reduce_dims.resize(tmp.dims().size());
    for (int i = 0; i < reduce_dims.size(); ++i) reduce_dims[i] = i;
    auto stream = context.cuda_device_context().stream();
    TensorReduce<bool, bool, BitwiseAdd, IdentityFunctor<bool>>(
        tmp, z, reduce_dims, true, BitwiseAdd(), IdentityFunctor<bool>(),
        stream);
  }
};

}  // namespace operators
}  // namespace paddle

#define REGISTER_COMPARE_REDUCE_CUDA_KERNEL(op_type, functor)          \
  REGISTER_OP_CUDA_KERNEL(                                             \
      op_type, paddle::operators::CompareReduceOpKernel<               \
                   paddle::platform::CUDADeviceContext, functor<int>>, \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<int64_t>>,      \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<float>>,        \
      paddle::operators::CompareReduceOpKernel<                        \
          paddle::platform::CUDADeviceContext, functor<double>>);
REGISTER_COMPARE_REDUCE_CUDA_KERNEL(equal_reduce,
                                    paddle::operators::EqualReduceFunctor);
