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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/diagonal_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, int INPUT_DIM_SIZE, int OUTPUT_DIM_SIZE>
__global__ void Diagonal(const T* input_data, T* output_data,
                         const int64_t offset_, int64_t axis1_, int64_t axis2_,
                         int64_t* input_stride, int64_t* output_stride,
                         int64_t numel) {
  CUDA_KERNEL_LOOP(idx, numel) {
    int64_t idx_dim[INPUT_DIM_SIZE] = {0};
    int64_t temp = 0;
    for (size_t i = 0; i < INPUT_DIM_SIZE - 1; i++) {
      idx_dim[i] = (idx - temp) / input_stride[i];
      temp = temp + idx_dim[i] * input_stride[i];
    }
    idx_dim[INPUT_DIM_SIZE - 1] = idx - temp;

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    int64_t out_dim[OUTPUT_DIM_SIZE] = {0};
    int temp_pos = 0;
    for (int i = 0; i < INPUT_DIM_SIZE; i++) {
      if (i != axis1_ && i != axis2_) {
        out_dim[temp_pos] = idx_dim[i];
        temp_pos++;
      }
    }
    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis2_dim;
      flag = true;
    }
    if (flag) {
      int64_t idx_output = 0;
      for (size_t i = 0; i < OUTPUT_DIM_SIZE - 1; i++) {
        idx_output = idx_output + out_dim[i] * output_stride[i];
      }
      idx_output = idx_output + out_dim[OUTPUT_DIM_SIZE - 1];
      output_data[idx_output] = input_data[idx];
    }
  }
}

template <typename T>
class DiagonalCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    int input_stride_size = input_dim_size - 1;
    int64_t* host_input_stride;
    host_input_stride =
        reinterpret_cast<int64_t*>(malloc(input_stride_size * sizeof(int64_t)));
    for (size_t i_input = 0; i_input < input_stride_size; i_input++) {
      int64_t temp_stride = 1;
      for (size_t j = i_input + 1; j < input_dim_size; j++) {
        temp_stride = temp_stride * input_dim[j];
      }
      host_input_stride[i_input] = temp_stride;
    }
    int64_t* input_stride;
    cudaMalloc(reinterpret_cast<void**>(&input_stride),
               input_stride_size * sizeof(int64_t));
    cudaMemcpy(reinterpret_cast<void*>(input_stride),
               reinterpret_cast<void*>(host_input_stride),
               input_stride_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int output_stride_size = output_dim_size - 1;
    int64_t* host_output_stride;
    host_output_stride = reinterpret_cast<int64_t*>(
        malloc(output_stride_size * sizeof(int64_t)));
    for (size_t i_output = 0; i_output < output_stride_size; i_output++) {
      int64_t temp_stride = 1;
      for (size_t j = i_output + 1; j < output_dim_size; j++) {
        temp_stride = temp_stride * output_dim[j];
      }
      host_output_stride[i_output] = temp_stride;
    }
    int64_t* output_stride;
    cudaMalloc(reinterpret_cast<void**>(&output_stride),
               output_stride_size * sizeof(int64_t));
    cudaMemcpy(reinterpret_cast<void*>(output_stride),
               reinterpret_cast<void*>(host_output_stride),
               output_stride_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;
    int64_t numel = input->numel();

    int threads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + threads - 1) / threads;

    switch (input_dim_size) {
      case 2:
        Diagonal<T, 2, 1><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 3:
        Diagonal<T, 3, 2><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 4:
        Diagonal<T, 4, 3><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 5:
        Diagonal<T, 5, 4><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 6:
        Diagonal<T, 6, 5><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 7:
        Diagonal<T, 7, 6><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 8:
        Diagonal<T, 8, 7><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      case 9:
        Diagonal<T, 9, 8><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 10, but received %d.",
            input_dim_size));
    }
    cudaFree(input_stride);
    cudaFree(output_stride);
    free(host_input_stride);
    free(host_output_stride);
  }
};

template <typename T, int DX_DIM_SIZE, int DOUT_DIM_SIZE>
__global__ void GradDiagonal(const T* dout_data, T* dx_data,
                             const int64_t offset_, int64_t axis1_,
                             int64_t axis2_, int64_t* dout_stride,
                             int64_t* dx_stride, int64_t numel) {
  CUDA_KERNEL_LOOP(idx, numel) {
    int64_t idx_dim[DX_DIM_SIZE] = {0};
    int64_t temp = 0;
    for (size_t i = 0; i < DX_DIM_SIZE - 1; i++) {
      idx_dim[i] = (idx - temp) / dx_stride[i];
      temp = temp + idx_dim[i] * dx_stride[i];
    }
    idx_dim[DX_DIM_SIZE - 1] = idx - temp;

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    int64_t dout_dim[DOUT_DIM_SIZE] = {0};
    int temp_pos = 0;
    for (int i = 0; i < DX_DIM_SIZE; i++) {
      if (i != axis1_ && i != axis2_) {
        dout_dim[temp_pos] = idx_dim[i];
        temp_pos++;
      }
    }
    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      dout_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      dout_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      dout_dim[temp_pos] = axis2_dim;
      flag = true;
    }
    if (flag) {
      int64_t idx_output = 0;
      for (size_t i = 0; i < DOUT_DIM_SIZE - 1; i++) {
        idx_output = idx_output + dout_dim[i] * dout_stride[i];
      }
      idx_output = idx_output + dout_dim[DOUT_DIM_SIZE - 1];
      dx_data[idx] = dout_data[idx_output];
    } else {
      dx_data[idx] = static_cast<T>(0);
    }
  }
}

template <typename T>
class DiagonalGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const auto* dout_data = dout->data<T>();
    auto dout_dim = dout->dims().Get();
    auto dout_dim_size = dout->dims().size();

    int dout_stride_size = dout_dim_size - 1;
    int64_t* host_dout_stride;
    host_dout_stride =
        reinterpret_cast<int64_t*>(malloc(dout_stride_size * sizeof(int64_t)));
    for (size_t i_dout = 0; i_dout < dout_stride_size; i_dout++) {
      int64_t temp_stride = 1;
      for (size_t j = i_dout + 1; j < dout_dim_size; j++) {
        temp_stride = temp_stride * dout_dim[j];
      }
      host_dout_stride[i_dout] = temp_stride;
    }
    int64_t* dout_stride;
    cudaMalloc(reinterpret_cast<void**>(&dout_stride),
               dout_stride_size * sizeof(int64_t));
    cudaMemcpy(reinterpret_cast<void*>(dout_stride),
               reinterpret_cast<void*>(host_dout_stride),
               dout_stride_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    auto* dx_data = dx->mutable_data<T>(context.GetPlace());
    auto dx_dim = dx->dims().Get();
    auto dx_dim_size = dx->dims().size();

    int dx_stride_size = dx_dim_size - 1;
    int64_t* host_dx_stride;
    host_dx_stride =
        reinterpret_cast<int64_t*>(malloc(dx_stride_size * sizeof(int64_t)));
    for (size_t i_dx = 0; i_dx < dx_stride_size; i_dx++) {
      int64_t temp_stride = 1;
      for (size_t j = i_dx + 1; j < dx_dim_size; j++) {
        temp_stride = temp_stride * dx_dim[j];
      }
      host_dx_stride[i_dx] = temp_stride;
    }
    int64_t* dx_stride;
    cudaMalloc(reinterpret_cast<void**>(&dx_stride),
               dx_stride_size * sizeof(int64_t));
    cudaMemcpy(reinterpret_cast<void*>(dx_stride),
               reinterpret_cast<void*>(host_dx_stride),
               dx_stride_size * sizeof(int64_t), cudaMemcpyHostToDevice);

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? dx_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? dx_dim_size + axis2 : axis2;

    int64_t numel = dx->numel();

    int threads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + threads - 1) / threads;

    switch (dx_dim_size) {
      case 2:
        GradDiagonal<T, 2, 1><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 3:
        GradDiagonal<T, 3, 2><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 4:
        GradDiagonal<T, 4, 3><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 5:
        GradDiagonal<T, 5, 4><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 6:
        GradDiagonal<T, 6, 5><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 7:
        GradDiagonal<T, 7, 6><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 8:
        GradDiagonal<T, 8, 7><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      case 9:
        GradDiagonal<T, 9, 8><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                                   axis1_, axis2_, dout_stride,
                                                   dx_stride, numel);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of output(input@Grad) should be less than 10, but "
            "received %d.",
            dx_dim_size));
    }
    cudaFree(dout_stride);
    cudaFree(dx_stride);
    free(host_dout_stride);
    free(host_dx_stride);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(diagonal, ops::DiagonalCUDAKernel<int>,
                        ops::DiagonalCUDAKernel<int64_t>,
                        ops::DiagonalCUDAKernel<float>,
                        ops::DiagonalCUDAKernel<double>);

REGISTER_OP_CUDA_KERNEL(diagonal_grad, ops::DiagonalGradCUDAKernel<int>,
                        ops::DiagonalGradCUDAKernel<int64_t>,
                        ops::DiagonalGradCUDAKernel<float>,
                        ops::DiagonalGradCUDAKernel<double>);
