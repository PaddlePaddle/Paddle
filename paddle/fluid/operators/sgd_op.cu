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
#include "paddle/fluid/operators/sgd_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

namespace {

template <typename T>
__global__ void SGDKernel(const T* g, const T* p, const T* learning_rate,
                          const int num, T* p_out) {
  T lr = learning_rate[0];
  int grid_size = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += grid_size) {
    T g_data = g[i];
    T p_data = p[i];
    p_out[i] = p_data - lr * g_data;
  }
}

template <typename T, int block_size>
__global__ void SparseSGDFunctorKernel(const T* selected_rows,
                                       const int64_t* rows,
                                       const T* learning_rate, T* tensor_out,
                                       int64_t row_numel) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  selected_rows += ty * row_numel;
  tensor_out += rows[ty] * row_numel;

  for (int index = tid; index < row_numel; index += block_size) {
    // Since index in rows of SelectedRows can be duplicate, we have to use
    // Atomic Operation to avoid concurrent write error.
    paddle::platform::CudaAtomicAdd(
        tensor_out + index, -1.0 * learning_rate[0] * selected_rows[index]);
  }
}
}  // namespace

template <typename T>
class SGDOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param = ctx.Input<framework::Tensor>("Param");
    auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    auto* grad_var = ctx.InputVar("Grad");
    // Actually, all tensors are LoDTensor except SelectedRows.
    if (grad_var->IsType<framework::LoDTensor>()) {
      param_out->mutable_data<T>(ctx.GetPlace());
      auto* grad = ctx.Input<framework::Tensor>("Grad");
      auto* grad_data = grad->data<T>();
      auto* param_data = param->data<T>();
      auto* param_out_data = param_out->data<T>();

      int block = 512;
      int grid = (param->numel() + block - 1) / block;

      SGDKernel<T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          grad_data, param_data, learning_rate->data<T>(), param->numel(),
          param_out_data);

    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
      // This manual optimization brings difficulty to track data dependency.
      // It's better to find a more elegant solution.
      PADDLE_ENFORCE_EQ(param, param_out);
      auto* grad = ctx.Input<framework::SelectedRows>("Grad");

      auto in_height = grad->height();
      auto out_dims = param_out->dims();
      PADDLE_ENFORCE_EQ(in_height, out_dims[0]);

      auto& in_value = grad->value();
      framework::Vector<int64_t> in_rows(grad->rows());

      int64_t in_row_numel = in_value.numel() / in_rows.size();
      PADDLE_ENFORCE_EQ(in_row_numel, param_out->numel() / in_height);

      auto* in_data = in_value.data<T>();
      auto* out_data = param_out->data<T>();

      const int block_size = 256;
      dim3 threads(block_size, 1);
      dim3 grid(1, in_rows.size());
      SparseSGDFunctorKernel<
          T, 256><<<grid, threads, 0, ctx.cuda_device_context().stream()>>>(
          in_data, in_rows.CUDAData(ctx.GetPlace()), learning_rate->data<T>(),
          out_data, in_row_numel);

    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sgd, ops::SGDOpCUDAKernel<float>,
                        ops::SGDOpCUDAKernel<double>);
