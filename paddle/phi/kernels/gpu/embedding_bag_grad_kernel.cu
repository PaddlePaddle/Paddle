// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/embedding_bag_grad_kernel.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

enum class CalMode_c { ksum, kmean, kmax};

// kernelfunc, calculate the grad of the variable 'weight'
template <typename T, typename IdT>
__global__ void EmbeddingBagWeightsGrad(const int output_dim,
                                        const IdT *input,
                                        const T *params,
                                        const T *grads,
                                        T *weights_grad,
                                        CalMode_c mode) {
  const int bag_idx = blockIdx.y;
  const int sequence_length = gridDim.y;
  const int bag_number = gridDim.x;
  const int sample_idx = blockIdx.x;
  const int paramsIdx =
      input[(sample_idx * sequence_length) + bag_idx] * output_dim;
  const int gradsIdx = sample_idx * output_dim;
  float partialDotProduct = 0.0f;
  for (int i = threadIdx.x; i < output_dim; i++) {
    partialDotProduct +=
        static_cast<float>(params[paramsIdx + i] * grads[gradsIdx + i]);
  }
  if (mode == CalMode_c::kmean) {
    partialDotProduct /= static_cast<float>(sequence_length);
  }

  if (threadIdx.x == 0) {
    weights_grad[(sample_idx * sequence_length) + bag_idx] =
        static_cast<T>(partialDotProduct);
  }
}
// asist in obtain the map between the indices and the rows of params
// can refer 'index_vec' in embedding_bag_grad_kernel.cc(in line 83)
// 
template <typename IdT>
__global__ void PrepTempArraysKernel(const IdT *indices,
                                     IdT *sortedIndices,
                                     IdT *sortedIndicesCounter,
                                     const int indices_size) {
  const int arrayIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (arrayIdx < indices_size) {
    sortedIndices[arrayIdx] = indices[arrayIdx];
    sortedIndicesCounter[arrayIdx] = arrayIdx;
  }
}

template <typename T, typename IdT>
__global__ void EmbeddingBagParamsGrad(const int output_dim,
                                       const int sequence_length,
                                       const IdT *sortedIndices,
                                       const IdT *counter,
                                       const T *weights,
                                       const T *grads,
                                       T *params_grad,
                                       CalMode_c mode) {
  const int sample_idx = blockIdx.x;
  const int bag_idx = blockIdx.y;
  const int feature_idx = threadIdx.x + bag_idx * blockDim.x;
  const int params_idx = __ldg(sortedIndices + sample_idx);
  // refer embeddingbag in tensorflow/addons, spin up a warp for each element
  // of the indices array, having each warp check the previous element, 
  // if the same, return without operations. If not, the warp iterates forward
  // and accumulates gradient. The operation is to avoid repeated reads and writes
  if (sample_idx > 0) {
    const int prev_idx = __ldg(sortedIndices + sample_idx - 1);
    if (prev_idx == params_idx) {
      return;
    }
  }
  int end_idx = sample_idx;
  while (end_idx < gridDim.x - 1) {
    int next_idx = end_idx + 1;
    int next_params_idx = __ldg(sortedIndices + next_idx);
    if (next_params_idx == params_idx) {
      end_idx += 1;
    } else {
      break;
    }
  }
  if (feature_idx < output_dim) {
    const int outputoffset = (params_idx * output_dim) + feature_idx;
    float accum = 0.0f;

    for (int i = sample_idx; i <= end_idx; ++i) {
      int indices_idx = __ldg(counter + i);
      auto weight_slice = weights[indices_idx];
      auto grad_slice =
          __ldg(grads + (indices_idx / sequence_length) + feature_idx);
      accum += static_cast<float>(weight_slice * grad_slice);
    }
    if (mode == CalMode_c::kmean) {
      accum /= static_cast<float>(sequence_length);
    }
    params_grad[outputoffset] = static_cast<T>(accum);
  }
}
template <typename T, typename Context>
struct EmbeddingBagGradCUDAFunctor {
  EmbeddingBagGradCUDAFunctor(const Context &dev_ctx,
                              const DenseTensor &input,
                              const DenseTensor &params,
                              const DenseTensor &weight,
                              const DenseTensor &out_grad,
                              const std::string &mode,
                              DenseTensor *params_grad,
                              DenseTensor *weight_grad)
      : dev_ctx_(dev_ctx),
        input_(input),
        params_(params),
        weight_(weight),
        out_grad_(out_grad),
        mode_(mode),
        params_grad_(params_grad),
        weight_grad_(weight_grad) {}

  template <typename IdT>
  void apply() {
    dev_ctx_.template Alloc<T>(params_grad_);
    dev_ctx_.template Alloc<T>(weight_grad_);

    const IdT *indices_d = input_.data<IdT>();
    const T *params_value_d = params_.data<T>();
    const T *weight_value_d = weight_.data<T>();
    const T *grad_d = out_grad_.data<T>();

    T *params_grad_d = params_grad_->data<T>();
    T *weight_grad_d = weight_grad_->data<T>();

    size_t output_dim = params_.dims()[1];
    size_t sequence_length = input_.dims()[1];

    const int kThreadsPerBlock = 32;
    const int bag_number = input_.dims()[0];

    dim3 grids(bag_number, sequence_length, 1);

    CalMode_c mode_enum = CalMode_c::ksum;
    if (mode_ == "mean") {
      CalMode_c mode_enum = CalMode_c::kmean;
    }

    EmbeddingBagWeightsGrad<T, IdT>
        <<<grids, kThreadsPerBlock, 0, dev_ctx_.stream()>>>(output_dim,
                                                            indices_d,
                                                            params_value_d,
                                                            grad_d,
                                                            weight_grad_d,
                                                            mode_enum);

    int allocsize = input_.dims()[0] * input_.dims()[1];

    auto sortedIndices =
        memory_utils::Alloc(dev_ctx_.GetPlace(), allocsize * sizeof(IdT *));
    IdT *sortedIndices_d = reinterpret_cast<IdT *>(sortedIndices->ptr());

    auto sortedIndicesCounter = memory_utils::Alloc(
        dev_ctx_.GetPlace(),
        allocsize * sizeof(IdT *),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));
    IdT *sortedIndicesCounter_d =
        reinterpret_cast<IdT *>(sortedIndicesCounter->ptr());

    const int indices_size = input_.dims()[0] * input_.dims()[1];
    const int params_size = params_.dims()[0] * params_.dims()[1];

    const int total_blocks = Eigen::divup(indices_size, kThreadsPerBlock);

    dim3 grids_2(total_blocks, 1, 1);

    // the target of these operations is to avoid parallel writes to the same element of 
    // the grads. So 'PrepTempArraysKernel' is designed to pre-sorting a copy of the indices(sourtedIndices),
    // and co-sorting a counter(sortedIndicesCounter).
    

    PrepTempArraysKernel<IdT>
        <<<grids_2, kThreadsPerBlock, 0, dev_ctx_.stream()>>>(
            indices_d, sortedIndices_d, sortedIndicesCounter_d, indices_size);

    thrust::device_ptr<IdT> sortedIndicesCounterDevicePtr(
        sortedIndicesCounter_d);
    thrust::device_ptr<IdT> sortedIndicesDevicePtr(sortedIndices_d);
    thrust::device_ptr<T> paramsGradDevicePtr(params_grad_d);

    thrust::fill(paramsGradDevicePtr,
                 paramsGradDevicePtr + static_cast<int>(params_size),
                 static_cast<T>(0.0f));
    thrust::sort_by_key(sortedIndicesDevicePtr,
                        sortedIndicesDevicePtr + indices_size,
                        sortedIndicesCounterDevicePtr);

    int threadsPerBlock;
    int blocksPerRow;
    if (output_dim <= 1024) {
      blocksPerRow = 1;
      threadsPerBlock = output_dim;
    } else {
      blocksPerRow = Eigen::divup(static_cast<int>(output_dim), 1024);
      // MAX_THREADS_PER_BLOCK
      threadsPerBlock =
          Eigen::divup(static_cast<int>(output_dim), blocksPerRow);
    }

    dim3 grids_3(indices_size, blocksPerRow, 1);
    EmbeddingBagParamsGrad<T, IdT>
        <<<grids_3, kThreadsPerBlock, 0, dev_ctx_.stream()>>>(
            output_dim,
            sequence_length,
            sortedIndices_d,
            sortedIndicesCounter_d,
            weight_value_d,
            grad_d,
            params_grad_d,
            mode_enum);
  }  // apply

 private:
  const phi::GPUContext &dev_ctx_;
  const DenseTensor &input_;
  const DenseTensor &params_;
  const DenseTensor &weight_;
  const DenseTensor &out_grad_;
  const std::string &mode_;
  DenseTensor *params_grad_;
  DenseTensor *weight_grad_;
};  // struct
template <typename T, typename Context>
void EmbeddingBagGradCUDAKernel(const Context &ctx,
                                const DenseTensor &input,
                                const DenseTensor &params,
                                const DenseTensor &weight,
                                const DenseTensor &out_grad,
                                const std::string &mode,
                                DenseTensor *params_grad,
                                DenseTensor *weight_grad) {
  EmbeddingBagGradCUDAFunctor<T, Context> functor(
      ctx, input, params, weight, out_grad, mode, params_grad, weight_grad);

  if (input.dtype() == phi::DataType::INT32) {
    functor.template apply<int>();
  } else if (input.dtype() == phi::DataType::INT64) {
    functor.template apply<int64_t>();
  } else if (input.dtype() == phi::DataType::INT16) {
    functor.template apply<int16_t>();
    PADDLE_THROW(phi::errors::Unimplemented(
        "emebddingbag input only support int16, int32 and int64"));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(embedding_bag_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::EmbeddingBagGradCUDAKernel,
                   float,
                   double) {}
