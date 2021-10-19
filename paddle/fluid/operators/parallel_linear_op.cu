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

#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/parallel_linear_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {
using framework::Tensor;

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

/*
    This function is to be called with one block per each column
*/
template <typename T>
__global__ void column_reduce(const T* matrix, T* result, int m /* lines */,
                              int n /* columns*/) {
  // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
  extern __shared__ unsigned char my_smem[];
  T* sdata = reinterpret_cast<T*>(my_smem);

  // normal tid
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // transposed tid for shared memory
  int new_tid = threadIdx.y + threadIdx.x * blockDim.y;

  // true x value in the matrix
  int real_x = threadIdx.x + blockDim.x * blockIdx.x;

  int i = real_x + n * threadIdx.y;
  const int it = n * blockDim.y;
  int offset = it;
  T accumulator = static_cast<T>(0);

  if (threadIdx.y < m && real_x < n) {
    // store all the values from this column in a warped way
    accumulator = matrix[i];
    while (i + offset < n * m) {
      accumulator += matrix[i + offset];
      offset += it;
    }
  }

  // save column reduction data in a transposed way
  sdata[new_tid] = accumulator;
  __syncthreads();

  for (size_t t = 16; t > 0; t >>= 1) {
    if (tid < 32 * 32 - 16) sdata[tid] += sdata[tid + t];
    __syncthreads();
  }

  if (threadIdx.y == 0 && real_x < n) result[real_x] = sdata[new_tid];
}

template <typename T>
__global__ void add_bias_kernel(T* data, const T* bias,
                                const int64_t batch_size, const int out_feat) {
  CUDA_KERNEL_LOOP(idx, batch_size * out_feat) {
    int col = idx % out_feat;
    paddle::platform::CudaAtomicAdd(&data[idx], bias[col]);
  }
}

template <typename T>
void add_bias(gpuStream_t stream, T* data, const T* bias,
              const int64_t batch_size, const int out_feat) {
  add_bias_kernel<<<GET_BLOCKS(batch_size * out_feat), CUDA_NUM_THREADS, 0,
                    stream>>>(data, bias, batch_size, out_feat);
}

// very slow to achieve bias grad, need to use reduce to speed!!
template <typename T>
__global__ void add_bias_grad_kernel(const T* dout_data, T* db_data,
                                     int batch_size, int out_feat) {
  CUDA_KERNEL_LOOP(idx, out_feat) {
    T temp = static_cast<T>(0);
    for (int i = 0; i < batch_size; ++i) {
      int select_indx = idx + out_feat * i;
      temp += dout_data[select_indx];
    }
    db_data[idx] += temp;
  }
}

template <typename T>
void add_bias_grad(gpuStream_t stream, const T* dout_data, T* db_data,
                   int batch_size, int out_feat) {
  add_bias_grad_kernel<<<GET_BLOCKS(out_feat), CUDA_NUM_THREADS, 0, stream>>>(
      dout_data, db_data, batch_size, out_feat);
}

template <typename DeviceContext, typename T>
class ParallelLinearCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    /* Tensor shape message
    X.dim = batch_size * in_dim
    W.dim = num_expert * in_dim * out_dim
    b.dim = num_expert * out_dim
    expert_count.dim = num_expert
    output.dim = batch_size * out_dim
    */

    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    auto* bias = ctx.Input<Tensor>("Bias");
    // auto* gpu_expert_count = ctx.Input<Tensor>("Expert_Count");
    auto expert_count = ctx.Attr<std::vector<int>>("expert_count");

    auto* output = ctx.Output<framework::LoDTensor>("Out");

    auto x_dims = x->dims();
    auto w_dims = w->dims();

    const auto num_expert = w_dims[0];
    const auto in_feat = w_dims[1];
    const auto out_feat = w_dims[2];
    const auto batch_size = x_dims[0];

    // get data ptr
    const T* x_data = x->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();

    const auto& dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();

    output->Resize({batch_size, out_feat});
    T* out_data = output->mutable_data<T>(ctx.GetPlace());

    // initialize
    // auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    // auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
    //  .eigen_device();
    // out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

    // ugly code to get expert_count in cpu
    // std::vector<int64_t> expert_count;
    // framework::TensorToVector(*gpu_expert_count, ctx.device_context(),
    // &expert_count);
    // dev_ctx.Wait();

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    for (int64_t i = 0, ptr = 0; i < num_expert; ++i) {
      if (expert_count[i] == 0) {
        continue;
      }
      // x_ptr[i] = x_data + ptr * in_feat;
      // w_ptr[i] = w_data + i * in_feat * out_feat;
      // out_ptr[i] = out_data + out_feat * ptr;

      blas.GEMM(CblasNoTrans, CblasNoTrans, expert_count[i], out_feat, in_feat,
                static_cast<T>(1), x_data + ptr * in_feat,
                w_data + i * in_feat * out_feat, static_cast<T>(0),
                out_data + out_feat * ptr);

      // TODO(shenliang03): speed up for moe
      add_bias<T>(dev_ctx.stream(), out_data + out_feat * ptr,
                  bias_data + i * out_feat, expert_count[i], out_feat);

      ptr += expert_count[i];
    }
  }
};

template <typename DeviceContext, typename T>
class ParallelLinearGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* w = ctx.Input<Tensor>("W");

    // auto* gpu_expert_count = ctx.Input<Tensor>("Expert_Count");
    auto expert_count = ctx.Attr<std::vector<int>>("expert_count");

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto x_dims = x->dims();
    auto w_dims = w->dims();

    const auto num_expert = w_dims[0];
    const auto in_feat = w_dims[1];
    const auto out_feat = w_dims[2];
    const auto batch_size = x_dims[0];

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();

    // initialize
    dx->mutable_data<T>(ctx.GetPlace());
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));

    dw->mutable_data<T>(ctx.GetPlace());
    auto dw_eigen = framework::EigenVector<T>::Flatten(*dw);
    dw_eigen.device(place) = dw_eigen.constant(static_cast<T>(0));

    // get data ptr
    const T* x_data = x->data<T>();
    const T* w_data = w->data<T>();
    const T* dout_data = dout->data<T>();
    T* dx_data = dx->data<T>();
    T* dw_data = dw->data<T>();

    db->mutable_data<T>(ctx.GetPlace());
    auto db_eigen = framework::EigenVector<T>::Flatten(*db);
    db_eigen.device(place) = db_eigen.constant(static_cast<T>(0));
    T* db_data = db->data<T>();

    // ugly code to get expert_count in cpu
    // std::vector<int64_t> expert_count;
    // framework::TensorToVector(*gpu_expert_count, ctx.device_context(),
    // &expert_count);
    // dev_ctx.Wait();

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);

    // bias
    dim3 block_threads(32, 32);
    dim3 grid_threads(out_feat / 32 + (out_feat % 32 ? 1 : 0), 1);

    for (int64_t i = 0, ptr = 0; i < num_expert; ++i) {
      if (expert_count[i] == 0) {
        continue;
      }

      // dx = dout_data * y^T
      blas.GEMM(CblasNoTrans, CblasTrans, expert_count[i], in_feat, out_feat,
                static_cast<T>(1), dout_data + ptr * out_feat,
                w_data + i * in_feat * out_feat, static_cast<T>(0),
                dx_data + in_feat * ptr);

      // dy = x^T * dout_data
      blas.GEMM(CblasTrans, CblasNoTrans, in_feat, out_feat, expert_count[i],
                static_cast<T>(1), x_data + in_feat * ptr,
                dout_data + ptr * out_feat, static_cast<T>(0),
                dw_data + i * in_feat * out_feat);

      // TODO(shenliang03): need use reduce_sum to optimize performance
      // add_bias_grad<T>(ctx.cuda_device_context().stream(),
      //                  dout_data + ptr * out_feat, db_data + i * out_feat,
      //                  expert_count[i], out_feat);

      column_reduce<<<grid_threads, block_threads, sizeof(T) * 1024,
                      ctx.cuda_device_context().stream()>>>(
          dout_data + ptr * out_feat, db_data + i * out_feat, expert_count[i],
          out_feat);

      ptr += expert_count[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

using GPUCtx = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(parallel_linear,
                        ops::ParallelLinearCUDAKernel<GPUCtx, float>,
                        ops::ParallelLinearCUDAKernel<GPUCtx, double>,
                        ops::ParallelLinearCUDAKernel<GPUCtx, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    parallel_linear_grad, ops::ParallelLinearGradOpCUDAKernel<GPUCtx, float>,
    ops::ParallelLinearGradOpCUDAKernel<GPUCtx, double>,
    ops::ParallelLinearGradOpCUDAKernel<GPUCtx, plat::float16>);
