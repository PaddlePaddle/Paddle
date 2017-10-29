/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/operators/sgd_op.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {

namespace {
template <typename T>
__global__ void SparseSGDFunctorKernel(const T* selected_rows,
                                       const int64_t* rows,
                                       const T* learning_rate, T* tensor_out,
                                       int64_t row_numel, int block_size) {
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
struct SparseSGDFunctor<platform::GPUPlace, T> {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input,
                  const framework::Tensor& learning_rate,
                  framework::Tensor* output) {
    auto in_height = input.height();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_EQ(in_height, out_dims[0]);

    auto& in_value = input.value();
    auto& in_rows = input.rows();

    int64_t in_row_numel = in_value.numel() / in_rows.size();
    PADDLE_ENFORCE_EQ(in_row_numel, output->numel() / in_height);

    auto* in_data = in_value.data<T>();
    auto* out_data = output->data<T>();

    int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid(1, in_rows.size());
    SparseSGDFunctorKernel<
        T><<<grid, threads, 0,
             reinterpret_cast<const platform::CUDADeviceContext&>(context)
                 .stream()>>>(in_data, in_rows.data(), learning_rate.data<T>(),
                              out_data, in_row_numel, block_size);
  }
};

template struct SparseSGDFunctor<platform::GPUPlace, float>;
template struct SparseSGDFunctor<platform::GPUPlace, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(sgd, ops::SGDOpKernel<paddle::platform::GPUPlace, float>,
                       ops::SGDOpKernel<paddle::platform::GPUPlace, double>);
