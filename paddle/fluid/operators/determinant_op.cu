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
void Determinant(const Tensor& input, const framework::ExecutionContext ctx, int rank, int batch_count, Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, ctx.device_context(),&input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_idx = input_vec.begin() + i * rank * rank;
      auto end_idx = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_idx,
                             end_idx);  // get every square matrix data
      Eigen::MatrixXf matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      output_vec.push_back(matrix.determinant());
    }
    framework::TensorFromVector(output_vec, output);
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

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }

    auto rank = input_dim[input_dim_size - 1];
    Determinant<T>(*input, context, rank, batch_count, output);
    auto output_dims = framework::slice_ddim(input->dims(), 0, input_dim_size - 2);
    output->Resize(output_dims);
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
void SlogDeterminant(const Tensor& input, const framework::ExecutionContext ctx, int rank, int batch_count,
  Tensor* output) {
    std::vector<T> input_vec;
    std::vector<float> sin_vec;
    std::vector<float> log_vec;
    std::vector<float> output_vec;
    framework::TensorToVector(input, ctx.device_context(), &input_vec);
    for (int i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_idx = input_vec.begin() + i * rank * rank;
      auto end_idx = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_idx,
                             end_idx);  // get every square matrix data
      Eigen::MatrixXf matrix(rank, rank);
      for (int i = 0; i < rank; ++i) {
        for (int j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      sin_vec.push_back(sign(matrix.determinant()));
      log_vec.push_back(log(matrix.determinant()));
    }
    // merge sin_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sin_vec.begin(), sin_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    framework::TensorFromVector(output_vec, output);
  }

template <typename T>
class SlogDeterminantCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    int batch_count = 1;
    for (int i = 0; i < input->dims().size() - 2; i++) {
      batch_count *= input_dim[i];
    }

    auto rank = input_dim[input_dim_size - 1];
    SlogDeterminant<T>(*input, context, rank, batch_count, output);
    std::vector<int> output_dim_vec(input_dim.begin(), input_dim.end() - 2);
    output_dim_vec.insert(output_dim_vec.begin(), 2); // make the output dims as same as numpy
    auto output_dims = framework::make_ddim(output_dim_vec);
    output->Resize(output_dims);
    VLOG(2) << "output dim:" << output->dims();
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
