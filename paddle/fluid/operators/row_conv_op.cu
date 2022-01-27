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

#include "paddle/fluid/operators/row_conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using framework::Tensor;

namespace {

inline int DivUp(int x, int y) { return (x + y - 1) / y; }

// Forward prop (shared memory version, for small future_context)
template <typename T>
__global__ void RowConvForwardSharedMemory(const T *in, const T *wt,
                                           int num_sequence, int input_dim,
                                           int future_context,
                                           const size_t *batch_indices,
                                           T *out) {
  int blx = blockDim.x;
  int bly = blockDim.y;
  int thx = threadIdx.x;
  int thy = threadIdx.y;
  int d = blockIdx.x * blx + thx;  // index along input dim

  extern __shared__ T mem[];
  T *sw = mem;

  if (thy < future_context) {
    sw[thy * blx + thx] =
        (d < input_dim) ? wt[thy * input_dim + d] : static_cast<T>(0);
  }
  __syncthreads();
  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
           w++) {
        sum += (d < input_dim)
                   ? sw[w * blx + thx] * in[(start + k + w) * input_dim + d]
                   : static_cast<T>(0);
      }
      if (d < input_dim) {
        out[(start + k) * input_dim + d] = sum;
      }
    }
  }
}

// Forward prop (naive version)
template <typename T>
__global__ void RowConvForward(const T *in, const T *wt, int num_sequence,
                               int input_dim, int future_context,
                               const size_t *batch_indices, T *out) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int bly = blockDim.y;
  int thy = threadIdx.y;

  if (d >= input_dim) return;
  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
           w++) {
        sum += (wt[w * input_dim + d] * in[(start + k + w) * input_dim + d]);
      }
      out[(start + k) * input_dim + d] = sum;
    }
  }
}

// Compute input gradient (shared memory version, for small future_context)
template <typename T>
__global__ void RowConvGradInputSharedMemory(const T *dout, const T *wt,
                                             int num_sequence, int input_dim,
                                             int future_context,
                                             const size_t *batch_indices,
                                             T *din) {
  int blx = blockDim.x;
  int bly = blockDim.y;
  int thx = threadIdx.x;
  int thy = threadIdx.y;
  int d = blockIdx.x * blx + thx;  // index along input dim

  extern __shared__ T mem[];
  T *sw = mem;
  if (thy < future_context) {
    sw[thy * blx + thx] =
        (d < input_dim) ? wt[thy * input_dim + d] : static_cast<T>(0);
  }
  __syncthreads();

  int current_timesteps = 0;
  for (int i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k - w) >= 0); w++) {
        sum += (d < input_dim)
                   ? (sw[w * blx + thx] * dout[(k + start - w) * input_dim + d])
                   : static_cast<T>(0);
      }
      if (d < input_dim) {
        din[(k + start) * input_dim + d] = sum;
      }
    }
  }
}

// Compute input gradient (Naive version)
template <typename T>
__global__ void RowConvGradInput(const T *dout, const T *wt, int num_sequence,
                                 int input_dim, int future_context,
                                 const size_t *batch_indices, T *din) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int bly = blockDim.y;
  int thy = threadIdx.y;

  if (d >= input_dim) return;
  int current_timesteps = 0;

  for (int i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    current_timesteps = end - start;

    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < future_context) && ((k - w) >= 0); w++) {
        sum += (wt[w * input_dim + d] * dout[(k + start - w) * input_dim + d]);
      }
      din[(k + start) * input_dim + d] = sum;
    }
  }
}

// Compute W gradient (small future_context version)
template <typename T>
__global__ void RowConvGradFilterImproved(const T *in, const T *dout,
                                          int num_sequence, int input_dim,
                                          int future_context, int block_x,
                                          int block_y,
                                          const size_t *batch_indices,
                                          T *dfilter) {
  int blx = blockDim.x;
  int bly = blockDim.y;
  int thx = threadIdx.x;
  int thy = threadIdx.y;
  int gx = blockIdx.x * blx;
  int d = gx + thx;  // index along input dim

  extern __shared__ T mem[];

  int xdim_sh_in = block_y;
  int xdim_sh_dout = block_y;
  int ydim_sh_in = block_x;
  int ydim_sh_dout = block_x + future_context - 1;
  int ydim_sh_dfilter = block_y;

  T *sh_in = mem;
  T *sh_dout = &mem[xdim_sh_in * ydim_sh_in];
  T *sh_dfilter = &mem[xdim_sh_in * ydim_sh_in + xdim_sh_dout * ydim_sh_dout];

  if (thy < future_context) {
    sh_dfilter[thy * ydim_sh_dfilter + thx] = static_cast<T>(0);
  }
  __syncthreads();

  // NOTE(zcd): temporary solution
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);

  for (int i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    int scaled_cur_steps =
        ((current_timesteps + block_x - 1) / block_x) * block_x;

    for (int k = thy; k < scaled_cur_steps; k += block_x) {
      int pos = start + k;
      sh_in[thx * ydim_sh_in + thy] =
          (d < input_dim && pos < end) ? in[pos * input_dim + d] : T(0);
      sh_dout[thx * ydim_sh_dout + thy + future_context - 1] =
          (d < input_dim && pos < end) ? dout[pos * input_dim + d] : T(0);
      __syncthreads();

      if (thy < future_context - 1) {
        int pos_offset = pos - future_context + 1;
        sh_dout[thx * ydim_sh_dout + thy] =
            (d < input_dim && pos_offset >= start)
                ? dout[pos_offset * input_dim + d]
                : T(0);
      }
      __syncthreads();

      for (int w = 0; w < future_context; w++) {
        T val = sh_in[thy * ydim_sh_in + thx] *
                sh_dout[thy * ydim_sh_dout + thx + future_context - 1 - w];
        __syncthreads();

        for (int offset = 16; offset > 0;
             offset = offset / 2) {  // blockDim.x is 32.
          val += platform::CudaShuffleDownSync(mask, val, offset);
        }
        __syncthreads();

        if (thx == 0) {
          sh_dfilter[w * ydim_sh_dfilter + thy] += val;
        }
        __syncthreads();
      }
    }
  }
  for (int w = thy; (w < future_context) && (d < input_dim); w += bly) {
    dfilter[w * input_dim + d] += sh_dfilter[w * ydim_sh_dfilter + thx];
  }
}

// Compute weight(filter) gradient
template <typename T>
__global__ void RowConvGradFilter(const T *in, const T *dout, int num_sequence,
                                  int input_dim, int future_context,
                                  int block_x, int block_y,
                                  const size_t *batch_indices, T *dfilter) {
  int blx = blockDim.x;
  int thx = threadIdx.x;
  int thy = threadIdx.y;
  int gx = blockIdx.x * blx;
  int d = gx + thx;  // index along input dim
  extern __shared__ T mem[];
  T *sh_in = mem;
  T *sh_dout = &mem[block_x * block_y];

  // NOTE(zcd): temporary solution
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;

    int scaled_cur_steps =
        ((current_timesteps + block_x - 1) / block_x) * block_x;

    for (int k = thy; k < scaled_cur_steps; k += block_x) {
      int pos = start + k;
      sh_in[thx * block_y + thy] =
          (d < input_dim && pos < end) ? in[pos * input_dim + d] : 0.0;
      __syncthreads();

      for (int w = 0; w < future_context; w++) {
        sh_dout[thx * block_y + thy] =
            (d < input_dim && (k - w) >= 0 && (k - w) < current_timesteps)
                ? dout[(pos - w) * input_dim + d]
                : 0.0;
        __syncthreads();

        T val = sh_in[thy * block_y + thx] * sh_dout[thy * block_y + thx];
        __syncthreads();

        for (int offset = 16; offset > 0;
             offset = offset / 2) {  // blockDim.x is 32.
          val += platform::CudaShuffleDownSync(mask, val, offset);
        }
        __syncthreads();

        if (thx == 0 && (gx + thy) < input_dim) {
          dfilter[w * input_dim + gx + thy] += val;
        }
      }
    }
  }
}

}  // namespace

template <typename T>
class RowConvKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<LoDTensor>("X");
    auto *Filter = context.Input<Tensor>("Filter");
    auto *Out = context.Output<LoDTensor>("Out");

    const T *in = X->data<T>();
    const T *weight = Filter->data<T>();
    T *out = Out->mutable_data<T>(context.GetPlace());
    bool is_tensor = X->lod().empty();
    int batch_size = 0;
    if (is_tensor) {
      batch_size = X->dims()[0];
    } else {
      batch_size = X->lod()[0].size() - 1;
    }
    int input_dim = 0;
    framework::Vector<size_t> batch_indices(batch_size + 1);
    int timesteps = X->dims()[1];
    if (is_tensor) {
      for (int i = 0; i < batch_size + 1; i++) {
        batch_indices[i] = i * timesteps;
      }
      input_dim = X->dims()[2];
    } else {
      batch_indices = X->lod()[0];
      input_dim = X->dims()[1];
    }

    int num_sequence = batch_indices.size() - 1;
    int future_context = Filter->dims()[0];
    size_t *idx = batch_indices.CUDAMutableData(context.GetPlace());
    auto stream = context.cuda_device_context().stream();

    if (future_context <= 32) {
      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
      int mem_per_block = (future_context * block_dim.x) * sizeof(T);
      RowConvForwardSharedMemory<
          T><<<grid_dim, block_dim, mem_per_block, stream>>>(
          in, weight, num_sequence, input_dim, future_context, idx, out);
    } else {
      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
      RowConvForward<T><<<grid_dim, block_dim, 0, stream>>>(
          in, weight, num_sequence, input_dim, future_context, idx, out);
    }
  }
};

template <typename T>
class RowConvGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<LoDTensor>("X");
    auto *Filter = context.Input<Tensor>("Filter");
    auto *dOut = context.Input<LoDTensor>(framework::GradVarName("Out"));
    const T *in = X->data<T>();
    const T *weights = Filter->data<T>();
    const T *dout = dOut->data<T>();

    Tensor *dX = context.Output<LoDTensor>(framework::GradVarName("X"));
    Tensor *dFilter = context.Output<Tensor>(framework::GradVarName("Filter"));
    int batch_size = 0;
    bool is_tensor = X->lod().empty();
    if (is_tensor) {
      batch_size = X->dims()[0];
    } else {
      batch_size = X->lod()[0].size() - 1;
    }

    int input_dim = 0;
    framework::Vector<size_t> batch_indices(batch_size + 1);
    int timesteps = X->dims()[1];
    if (is_tensor) {
      for (int i = 0; i < batch_size + 1; i++) {
        batch_indices[i] = i * timesteps;
      }
      input_dim = X->dims()[2];
    } else {
      batch_indices = X->lod()[0];
      input_dim = X->dims()[1];
    }
    // int input_dim = X->dims()[1];
    int num_sequence = batch_indices.size() - 1;
    int future_context = Filter->dims()[0];
    size_t *idx = batch_indices.CUDAMutableData(context.GetPlace());

    auto &device_ctx = context.cuda_device_context();
    pten::funcs::SetConstant<platform::CUDADeviceContext, T> zero;

    if (dFilter) {
      T *dfilter = dFilter->mutable_data<T>(context.GetPlace());
      zero(device_ctx, dFilter, static_cast<T>(0.0));

      if (future_context <= 32) {
        dim3 block_dim = dim3(32, 32);
        dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
        int block_x = block_dim.x;
        int block_y = block_dim.y;
        int mem_per_block =
            (block_y * block_x + block_y * (block_x + future_context - 1) +
             future_context * block_y) *
            sizeof(T);
        RowConvGradFilterImproved<
            T><<<grid_dim, block_dim, mem_per_block, device_ctx.stream()>>>(
            in, dout, num_sequence, input_dim, future_context, block_x, block_y,
            idx, dfilter);
      } else {
        dim3 block_dim = dim3(32, 32);
        dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
        int block_x = block_dim.x;
        int block_y = block_dim.y;
        int mem_per_block =
            (block_x * block_y * 2) * sizeof(T);  // For 2 arrays of size 32x32
        RowConvGradFilter<
            T><<<grid_dim, block_dim, mem_per_block, device_ctx.stream()>>>(
            in, dout, num_sequence, input_dim, future_context, block_x, block_y,
            idx, dfilter);
      }
    }

    if (dX) {
      T *din = dX->mutable_data<T>(context.GetPlace());
      if (future_context <= 32) {
        dim3 block_dim = dim3(32, 32);
        dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
        int mem_per_block = (future_context * block_dim.x) * sizeof(T);
        RowConvGradInputSharedMemory<
            T><<<grid_dim, block_dim, mem_per_block, device_ctx.stream()>>>(
            dout, weights, num_sequence, input_dim, future_context, idx, din);
      } else {
        dim3 block_dim = dim3(32, 32);
        dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
        RowConvGradInput<T><<<grid_dim, block_dim, 0, device_ctx.stream()>>>(
            dout, weights, num_sequence, input_dim, future_context, idx, din);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    row_conv, ops::RowConvKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    row_conv_grad,
    ops::RowConvGradKernel<paddle::platform::CUDADeviceContext, float>);
