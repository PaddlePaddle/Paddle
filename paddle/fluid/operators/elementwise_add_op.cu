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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/elementwise_add_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void ElementwiseAddCUDAKernel(const T *x, const T *y, T *z, int n,
                                         int post, int size) {
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx_x < size) {
    int idx_y = idx_x / post - (idx_x / (n * post)) * n;
    z[idx_x] = x[idx_x] + y[idx_y];
  }
}

template <typename T>
class ElementwiseAddKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using Tensor = framework::Tensor;

    const auto x = ctx.Input<Tensor>("X");
    const auto y = ctx.Input<Tensor>("Y");
    auto z = ctx.Output<Tensor>("Out");
    auto *z_data = z->mutable_data<T>(ctx.GetPlace());

    auto &device = *(ctx.cuda_device_context().eigen_device());
    const framework::DDim &x_dim = x->dims();
    framework::DDim y_dim = y->dims();
    int size = x->numel();
    if (x_dim == y_dim) {
      auto dim = framework::make_ddim({size});
      auto z_eigen = framework::EigenTensor<T, 1>::From(*z, dim);
      auto x_eigen = framework::EigenTensor<T, 1>::From(*x, dim);
      auto y_eigen = framework::EigenTensor<T, 1>::From(*y, dim);
      z_eigen.device(device) = x_eigen + y_eigen;
    } else {
      int axis = ctx.Attr<int>("axis");
      axis = (axis == -1 ? x_dim.size() - y_dim.size() : axis);
      y_dim = trim_trailing_singular_dims(y_dim);
      axis = (y_dim.size() == 0) ? x_dim.size() : axis;
      int pre, n, post;
      get_mid_dims(x_dim, y_dim, axis, &pre, &n, &post);
      int threads = 512;
      int grids = (size + threads - 1) / threads;
      auto stream = ctx.cuda_device_context().stream();
      ElementwiseAddCUDAKernel<T><<<grids, threads, 0, stream>>>(
          x->data<T>(), y->data<T>(), z_data, n, post, size);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    elementwise_add, ops::ElementwiseAddKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, int64_t>,
    ops::ElementwiseAddKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, float>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, double>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int>,
    ops::ElementwiseAddGradKernel<plat::CUDADeviceContext, int64_t>);
