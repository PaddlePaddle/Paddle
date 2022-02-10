/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/triangular_solve_op.h"

namespace paddle {
namespace operators {

template <typename T>
class MatrixReduceSumFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const Tensor& in, Tensor* out,
                  const framework::ExecutionContext& ctx) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<std::int64_t> in_dims = framework::vectorize(in.dims());
    auto in_size = in_dims.size();
    const std::vector<std::int64_t> out_dims =
        framework::vectorize(out->dims());
    auto out_size = out_dims.size();

    std::vector<std::int64_t> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(), out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);

    std::vector<int> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(idx);
      }
    }
    gpuStream_t stream = ctx.cuda_device_context().stream();
    TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        ctx.cuda_device_context(), in, out, kps::IdentityFunctor<T>(),
        out_reduce_dims, stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    triangular_solve,
    ops::TriangularSolveKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TriangularSolveKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    triangular_solve_grad,
    ops::TriangularSolveGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TriangularSolveGradKernel<paddle::platform::CUDADeviceContext,
                                   double>);
