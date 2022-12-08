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
#include "paddle/fluid/operators/cvm_op.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void CvmComputeKernel(const bool use_cvm,
                                 const int64_t item_width,
                                 const T* X,
                                 T* Y,
                                 int64_t numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    if (use_cvm) {
      if (i % item_width == 0) {
        Y[i] = log(X[i] + 1);
      } else if (i % item_width == 1) {
        Y[i] = log(X[i] + 1) - log(X[i - 1] + 1);
      } else {
        Y[i] = X[i];
      }
    } else {
      Y[i] = X[i / (item_width - 2) * item_width + i % (item_width - 2) + 2];
    }
  }
}

template <typename T>
__global__ void CvmGradComputeKernel(const bool use_cvm,
                                     const int64_t item_width,
                                     const T* CVM,
                                     const T* DY,
                                     T* DX,
                                     bool has_lod,
                                     const size_t* lod,
                                     size_t lod_size,
                                     int64_t numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    int offset = i % item_width;
    if (offset <= 1) {
      int cvm_id = i / item_width;
      if (has_lod) {
        int low = 1;
        int high = lod_size - 1;
        while (low < high) {
          int mid = (low + high) / 2;
          if (cvm_id < lod[mid])
            high = mid;
          else
            low = mid + 1;
        }
        cvm_id = low - 1;
      }
      DX[i] = CVM[2 * cvm_id + offset];
    } else {
      if (use_cvm) {
        DX[i] = DY[i];
      } else {
        DX[i] = DY[i / item_width * (item_width - 2) + i % item_width - 2];
      }
    }
  }
}

template <typename T>
class CVMCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<phi::DenseTensor>("X");
    const T* x_data = x->data<T>();

    auto batch_size = x->dims()[0];
    auto numel = x->numel();
    auto item_size = numel / batch_size;
    auto use_cvm = context.Attr<bool>("use_cvm");

    auto* y = context.Output<phi::DenseTensor>("Y");
    T* y_data = y->mutable_data<T>(context.GetPlace());

    // for Input X do not have Lod Information.
    auto stream = context.template device_context<phi::GPUContext>().stream();
    if (x->NumLevels() == 0) {
      CvmComputeKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                             PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS,
                         0,
                         stream>>>(
          use_cvm, item_size, x_data, y_data, y->numel());
    } else {
      auto lod = x->lod()[0];
      PADDLE_ENFORCE_EQ(
          batch_size,
          lod[lod.size() - 1],
          platform::errors::PreconditionNotMet(
              "Input(X)'s dim[0] must be equal to last element of lod"));
      CvmComputeKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                             PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS,
                         0,
                         stream>>>(
          use_cvm, item_size, x_data, y_data, y->numel());
    }
  }
};

template <typename T>
class CVMGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dx = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    const phi::DenseTensor* cvm = context.Input<phi::DenseTensor>("CVM");
    const T* cvm_data = cvm->data<T>();

    const auto* dOut =
        context.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const T* dout_data = dOut->data<T>();

    auto use_cvm = context.Attr<bool>("use_cvm");

    auto offset = 2;
    auto batch_size = dx->dims()[0];
    auto dx_numel = dx->numel();
    auto item_size = dx_numel / batch_size;

    // for Input X do not have Lod Information.
    auto stream = context.template device_context<phi::GPUContext>().stream();
    if (dx->NumLevels() == 0) {
      CvmGradComputeKernel<<<(dx_numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                 PADDLE_CUDA_NUM_THREADS,
                             PADDLE_CUDA_NUM_THREADS,
                             0,
                             stream>>>(use_cvm,
                                       item_size,
                                       cvm_data,
                                       dout_data,
                                       dx_data,
                                       false,
                                       NULL,
                                       0,
                                       dx_numel);
    } else {
      auto lod = dx->lod()[0];
      PADDLE_ENFORCE_EQ(
          batch_size,
          lod[lod.size() - 1],
          platform::errors::PreconditionNotMet(
              "Output(X@GRAD)'s dim[0] must be equal to last element of lod"));
      paddle::framework::MixVector<size_t> mixv_lod(&lod);
      CvmGradComputeKernel<<<(dx_numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                 PADDLE_CUDA_NUM_THREADS,
                             PADDLE_CUDA_NUM_THREADS,
                             0,
                             stream>>>(use_cvm,
                                       item_size,
                                       cvm_data,
                                       dout_data,
                                       dx_data,
                                       true,
                                       mixv_lod.CUDAData(context.GetPlace()),
                                       lod.size(),
                                       dx_numel);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cvm,
                        ops::CVMCUDAKernel<float>,
                        ops::CVMCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(cvm_grad,
                        ops::CVMGradCUDAKernel<float>,
                        ops::CVMGradCUDAKernel<double>);
