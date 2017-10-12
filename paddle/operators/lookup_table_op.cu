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

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTable(T* output, const T* table, const int32_t* ids,
                            const int N, const int K, const int D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int id = ids[idy];
    PADDLE_ASSERT(id >= 0);
    PADDLE_ASSERT(id < N);
    T* out = output + idy * D;
    const T* tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      out[i] = tab[i];
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T, int BlockDimX, int BlockDimY, int GridDimX>
__global__ void LookupTableGrad(T* table, const T* output, const int32_t* ids,
                                const int N, const int K, const int D) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * GridDimX;

  while (idy < K) {
    int id = ids[idy];
    PADDLE_ASSERT(id >= 0);
    PADDLE_ASSERT(id < N);
    const T* out = output + idy * D;
    T* tab = table + id * D;
    for (int i = idx; i < D; i += BlockDimX) {
      paddle::platform::CudaAtomicAdd(&tab[i], out[i]);
    }
    idy += BlockDimY * GridDimX;
  }
}

template <typename T>
class LookupTableCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto table_t = context.Input<Tensor>("W");
    auto ids_t = context.Input<Tensor>("Ids");
    auto output_t = context.Output<Tensor>("Out");

    size_t N = table_t->dims()[0];
    size_t D = table_t->dims()[1];
    size_t K = ids_t->numel();
    auto ids = ids_t->data<int32_t>();
    auto table = table_t->data<T>();
    auto output = output_t->mutable_data<T>(context.GetPlace());

    dim3 threads(128, 8);
    dim3 grids(8, 1);
    LookupTable<T, 128, 8, 8><<<
        grids, threads, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               context.device_context())
                               .stream()>>>(output, table, ids, N, K, D);
  }
};

template <typename T>
class LookupTableGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ids_t = context.Input<Tensor>("Ids");
    auto d_output_t = context.Input<Tensor>(framework::GradVarName("Out"));
    auto d_table_t = context.Output<Tensor>(framework::GradVarName("W"));

    int N = d_table_t->dims()[0];
    int D = d_table_t->dims()[1];
    int K = ids_t->numel();
    const int32_t* ids = ids_t->data<int32_t>();
    const T* d_output = d_output_t->data<T>();
    T* d_table = d_table_t->mutable_data<T>(context.GetPlace());

    auto t = framework::EigenVector<T>::Flatten(*d_table_t);
    t.device(context.GetEigenDevice<platform::GPUPlace>()) =
        t.constant(static_cast<T>(0));

    dim3 threads(128, 8);
    dim3 grids(8, 1);
    LookupTableGrad<T, 128, 8, 8><<<
        grids, threads, 0, reinterpret_cast<const platform::CUDADeviceContext&>(
                               context.device_context())
                               .stream()>>>(d_table, d_output, ids, N, K, D);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(lookup_table, ops::LookupTableCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(lookup_table_grad,
                       ops::LookupTableGradCUDAKernel<float>);
