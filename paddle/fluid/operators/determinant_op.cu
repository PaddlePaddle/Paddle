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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/determinant_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;

template <typename T>
__global__ void Determinant(const size_t numel, const T* in, int rank, T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numel) {
    Eigen::MatrixXf matrix(rank, rank);

    for (int i = 0; i < rank; ++i) {
      for (int j = 0; j < rank; ++j) {
        matrix(i, j) = in[rank * i + j];
      }
      out[tid] = matrix.determinant();
    }
  }
}

template <typename T>
__global__ void DeterminantGrad(const size_t numel, T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numel) {
    out[tid] = static_cast<T>(1);
  }
}
template <typename T>
class DeterminantCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();
    auto numel = output->numel();

    int threads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + threads - 1) / threads;

    auto rank = input_dim[input_dim_size - 1];
    Determinant<T><<<blocks, threads>>>(numel, input_data, rank, output_data);
  }
};

template <typename T>
class DeterminantGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const T* dout_data = dout->data<T>();
    auto dout_dim = vectorize(dout->dims());

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dx->numel();
    for (int64_t idx = 0; idx < numel; idx++) {
      dx_data[idx] = static_cast<T>(1);
    }
  }
};

template <typename T>
__global__ void SlogDeterminant(const size_t total, const T* in, int rank,
                                T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < total) {
    Eigen::MatrixXf matrix(rank, rank);

    for (int i = 0; i < rank; ++i) {
      for (int j = 0; j < rank; ++j) {
        matrix(i, j) = ingit[rank * i + j];
      }
      out[tid] = sin(matrix.determinant());
      out[tid + total] = log(matrix.determinant());
    }
  }
}

template <typename T>
class SlogDeterminantCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int threads = PADDLE_CUDA_NUM_THREADS;
    auto numel = output->numel() / 2;
    int blocks = (numel + threads - 1) / threads;

    auto rank = input_dim[input_dim_size - 1];
    SlogDeterminant<T><<<blocks, threads>>>(numel, input_data, rank,
                                            output_data);
  }
};

template <typename T>
class SlogDeterminantGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int threads = PADDLE_CUDA_NUM_THREADS;
    auto numel = output->numel() / 2;
    int blocks = (numel + threads - 1) / threads;

    auto rank = input_dim[input_dim_size - 1];
    DeterminantGrad<T><<<blocks, threads>>>(numel, output_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(determinant, ops::DeterminantCUDAKernel<int>,
                        ops::DeterminantCUDAKernel<int64_t>,
                        ops::DeterminantCUDAKernel<float>,
                        ops::DeterminantCUDAKernel<double>,
                        ops::DeterminantCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(determinant_grad, ops::DeterminantGradCUDAKernel<int>,
                        ops::DeterminantGradCUDAKernel<int64_t>,
                        ops::DeterminantGradCUDAKernel<float>,
                        ops::DeterminantGradCUDAKernel<double>);

REGISTER_OP_CUDA_KERNEL(slogdeterminant, ops::SlogDeterminantCUDAKernel<int>,
                        ops::SlogDeterminantCUDAKernel<int64_t>,
                        ops::SlogDeterminantCUDAKernel<float>,
                        ops::SlogDeterminantCUDAKernel<double>,
                        ops::SlogDeterminantCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(slogdeterminant_grad,
                        ops::DeterminantGradCUDAKernel<int>,
                        ops::SlogDeterminantGradCUDAKernel<int64_t>,
                        ops::SlogDeterminantGradCUDAKernel<float>,
                        ops::SlogDeterminantGradCUDAKernel<double>);
