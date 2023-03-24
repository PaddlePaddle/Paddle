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

#pragma once
#include "paddle/fluid/operators/unzip_op.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
namespace paddle {
namespace operators {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename LodType>
__global__ void unzipKernel(
    const T* X, const LodType* lod, T* Y, size_t col_size, size_t n) {
  CUDA_KERNEL_LOOP(i, n) {
    int lod_idx = i / col_size;
    if ((lod[lod_idx + 1] - lod[lod_idx]) > 0) {
      assert((lod[lod_idx + 1] - lod[lod_idx]) == col_size);
      int x_idx = 0;
      for (int j = 0; j < lod_idx; ++j) {
        if ((lod[j + 1] - lod[j]) > 0) {
          x_idx++;
        }
      }
      Y[i] = X[x_idx * col_size + (i % col_size)];
    } else {
      Y[i] = 0;
    }
  }
}

template <typename T, typename LodType>
class unzipCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<phi::DenseTensor>("X");
    const T* x_data = x->data<T>();

    const auto* lod = context.Input<phi::DenseTensor>("lod");
    const LodType* lod_data = lod->data<LodType>();

    auto col_size = x->dims()[1];
    auto row_size = lod->dims()[0] - 1;
    auto y_numel = col_size * row_size;

    auto* y = context.Output<phi::DenseTensor>("Y");
    T* y_data = y->mutable_data<T>(context.GetPlace());

    // for Input X do not have lod Information.
    auto stream = context.template device_context<phi::GPUContext>().stream();
    unzipKernel<<<(y_numel + PADDLE_CUDA_NUM_THREADS - 1) /
                      PADDLE_CUDA_NUM_THREADS,
                  PADDLE_CUDA_NUM_THREADS,
                  0,
                  stream>>>(x_data, lod_data, y_data, col_size, y_numel);
  }
};

template <typename T>
class unzipGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_THROW(phi::errors::Unimplemented("unzip_grad is unimplemented"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    unzip,
    ops::unzipCUDAKernel<float, int>,
    ops::unzipCUDAKernel<double, int>,
    ops::unzipCUDAKernel<paddle::platform::float16, int>,
    ops::unzipCUDAKernel<int, int>,
    ops::unzipCUDAKernel<bool, int>,
    ops::unzipCUDAKernel<int64_t, int>,
    ops::unzipCUDAKernel<float, int64_t>,
    ops::unzipCUDAKernel<double, int64_t>,
    ops::unzipCUDAKernel<paddle::platform::float16, int64_t>,
    ops::unzipCUDAKernel<int, int64_t>,
    ops::unzipCUDAKernel<bool, int64_t>,
    ops::unzipCUDAKernel<int64_t, int64_t>);

REGISTER_OP_CUDA_KERNEL(unzip_grad,
                        ops::unzipGradCUDAKernel<float>,
                        ops::unzipGradCUDAKernel<double>,
                        ops::unzipGradCUDAKernel<paddle::platform::float16>,
                        ops::unzipGradCUDAKernel<int>,
                        ops::unzipGradCUDAKernel<bool>,
                        ops::unzipGradCUDAKernel<int64_t>);
