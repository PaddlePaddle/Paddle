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
    int len = lod[lod_idx + 1] - lod[lod_idx];
    if (i >= lod_idx * col_size + len) {
      Y[i] = 0;
    } else {
      Y[i] = X[lod[lod_idx] + i % col_size];
    }
  }
}

template <typename T, typename DeviceContext, typename LodType = int64_t>
class unzipCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<phi::DenseTensor>("X");
    const T* x_data = x->data<T>();

    const auto* lod = context.Input<phi::DenseTensor>("lod");
    const LodType* lod_data = lod->data<LodType>();

    auto col_size = context.Attr<int>("len");
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

template <typename T, typename DeviceContext>
class unzipGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_THROW(phi::errors::Unimplemented("unzip_grad is unimplemented"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
PD_REGISTER_STRUCT_KERNEL(unzip,
                          GPU,
                          ALL_LAYOUT,
                          ops::unzipCUDAKernel,
                          float,
                          double,
                          plat::float16,
                          bool,
                          int,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
PD_REGISTER_STRUCT_KERNEL(unzip_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::unzipGradCUDAKernel,
                          float,
                          double,
                          plat::float16,
                          bool,
                          int,
                          int64_t,
                          phi::dtype::complex<float>,
                          phi::dtype::complex<double>) {}
