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

#include <typeinfo>
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/warpctc_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
void PrintTensor(const framework::LoDTensor& src,
                 const framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < static_cast<int>(vec.size()); ++i) {
    VLOG(3) << "vec[" << i << "] : " << vec[i];
  }
}

template <typename T>
__global__ void ReduceSumKernel(const T* d_in, T* d_out) {
  // Allocate shared memory
  extern __shared__ int partial_sum[];

  // Calculate thread ID
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load elements into shared memory
  partial_sum[threadIdx.x] = d_in[tid];
  __syncthreads();

  // Start at 1/2 block stride and divide by two each iteration
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    // Each thread does work unless it is further than the stride
    if (threadIdx.x < s) {
      partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Let the thread 0 for this block write it's result to main memory
  // Result is inexed by this block
  if (threadIdx.x == 0) {
    d_out[blockIdx.x] = partial_sum[0];
  }
}

template <typename T>
__global__ void CTCGradScaleKernel(T* d_out, const T* d_ctc, const T* d_loss,
                                   int scale, int Tmax, int B, int D) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elems = Tmax * B * D;
  int b_idx = (tid / D) % B;
  for (; tid < n_elems; tid += gridDim.x * blockDim.x) {
    d_out[tid] = d_ctc[tid] * d_loss[b_idx] / static_cast<T>(scale);
  }
}

template <typename T>
__global__ void CTCGradScaleKernel(T* d_out, const T* d_ctc, const T* d_loss,
                                   int64_t* scale, int Tmax, int B, int D) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elems = Tmax * B * D;
  int b_idx = (tid / D) % B;
  for (; tid < n_elems; tid += gridDim.x * blockDim.x) {
    d_out[tid] = d_ctc[tid] * d_loss[b_idx] / static_cast<T>(scale[0]);
  }
}

template <typename T>
__global__ void CTCGradBatchScaleKernel(T* d_out, const T* d_ctc,
                                        const T* d_loss, const int64_t* scales,
                                        int Tmax, int B, int D) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n_elems = Tmax * B * D;
  int b_idx = (tid / D) % B;
  // scale is vector, (B)
  for (; tid < n_elems; tid += gridDim.x * blockDim.x) {
    d_out[tid] = d_ctc[tid] * d_loss[b_idx] / scales[b_idx];
  }
}

template <typename DeviceContext, typename T>
class WarpCTCGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<LoDTensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));
    const Tensor* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    logits_grad->mutable_data<T>(ctx.GetPlace());
    bool norm_by_times = ctx.Attr<bool>("norm_by_times");
    bool norm_by_batchsize = ctx.Attr<bool>("norm_by_batchsize");
    bool norm_by_total_logits_len = ctx.Attr<bool>("norm_by_total_logits_len");

    if ((norm_by_times && norm_by_batchsize) ||
        (norm_by_times && norm_by_total_logits_len) ||
        (norm_by_batchsize && norm_by_total_logits_len)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "[warpctc grad] norm_by_times, norm_by_batchsize and "
          "norm_by_total_logits_len "
          "should one be true."));
    }

    if (ctx.HasInput("LogitsLength")) {
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      auto stream = dev_ctx.stream();
      int max_seq_length = warpctc_grad->dims()[0];  // Tmax
      int num_sequences = warpctc_grad->dims()[1];   // B
      int seq_width = warpctc_grad->dims()[2];       // D

      auto* logits_length = ctx.Input<framework::Tensor>("LogitsLength");
      const int64_t* logits_length_ptr = logits_length->data<int64_t>();

      int n_elems = max_seq_length * num_sequences * seq_width;
      int num_blocks =
          (n_elems + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
      int shm_bytes = PADDLE_CUDA_NUM_THREADS * sizeof(T);

      auto logits_grad_ptr =
          logits_grad->mutable_data<T>(ctx.GetPlace());  // (Tmax, B, D)
      auto warpctc_grad_ptr = warpctc_grad->data<T>();   // (Tmax, B, D)
      auto loss_grad_ptr = loss_grad->data<T>();         // (B, 1)

      if (norm_by_total_logits_len) {
        VLOG(3) << "norm_by_total_logits_len no impl ";
        // total length
        Tensor total_length;
        int64_t* total_length_ptr =
            total_length.mutable_data<int64_t>({1}, ctx.GetPlace());
        int bytes = num_sequences * sizeof(int64_t);
        ReduceSumKernel<int64_t><<<1, num_sequences, bytes, stream>>>(
            logits_length_ptr, total_length_ptr);

        CTCGradScaleKernel<
            T><<<num_blocks, PADDLE_CUDA_NUM_THREADS, shm_bytes, stream>>>(
            logits_grad_ptr, warpctc_grad_ptr, loss_grad_ptr, total_length_ptr,
            max_seq_length, num_sequences, seq_width);

      } else if (norm_by_batchsize) {
        VLOG(3) << "norm_by_batchsize ";
        CTCGradScaleKernel<
            T><<<num_blocks, PADDLE_CUDA_NUM_THREADS, shm_bytes, stream>>>(
            logits_grad_ptr, warpctc_grad_ptr, loss_grad_ptr, num_sequences,
            max_seq_length, num_sequences, seq_width);
      } else if (norm_by_times) {
        VLOG(3) << "norm_by_times ";
        CTCGradBatchScaleKernel<
            T><<<num_blocks, PADDLE_CUDA_NUM_THREADS, shm_bytes, stream>>>(
            logits_grad_ptr, warpctc_grad_ptr, loss_grad_ptr, logits_length_ptr,
            max_seq_length, num_sequences, seq_width);
      } else {
        VLOG(3) << "default ";
        CTCGradScaleKernel<
            T><<<num_blocks, PADDLE_CUDA_NUM_THREADS, shm_bytes, stream>>>(
            logits_grad_ptr, warpctc_grad_ptr, loss_grad_ptr, 1, max_seq_length,
            num_sequences, seq_width);
      }
    } else {
      math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *warpctc_grad,
          logits_grad, -1, 0, norm_by_times, norm_by_batchsize,
          norm_by_total_logits_len, math::kLengthBatchWidth);

      const T* loss_grad_data = loss_grad->data<T>();
      math::ScaleLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), loss_grad_data,
          logits_grad);
    }
  }
};

}  // operators
}  // paddle

namespace ops = paddle::operators;

// register forward and backward of CUDA OP must in same *.cu file.
// Eigen can be used on GPU device, but must be in *.cu file not *.cu.cc file.
// *.cu.cc also using GCC compiler. *.cu using NVCC compiler
REGISTER_OP_CUDA_KERNEL(
    warpctc, ops::WarpCTCKernel<paddle::platform::CUDADeviceContext, float>,
    ops::WarpCTCKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    warpctc_grad,
    ops::WarpCTCGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::WarpCTCGradCUDAKernel<paddle::platform::CUDADeviceContext, double>);
