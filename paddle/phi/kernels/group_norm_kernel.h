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

#pragma once

#include <string>

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/dense_tensor.h"
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#endif
#include <stdint.h>

namespace phi {

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* variance);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T, typename AccT = T>
class GroupNormDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T* input,
                  std::vector<int> input_shape,
                  const T* bias,
                  const T* scale,
                  AccT* temp_variance,
                  int groups,
                  float eps,
                  T* output,
                  AccT* mean,
                  AccT* variance,
                  const DataLayout data_layout);
};
#endif

template <typename T>
struct GroupNormNDHWCParams {
  // The output buffer. Layout NDHWC.
  T* dst;
  // The output buffer. Layout NDHWC.
  T* eleOut;
  // The input buffer. Layout NDHWC.
  T const* srcX;
  // The input buffer. Layout NDHWC.
  T const* srcY;
  // The input buffer. Layout NDHWC.
  T const* srcR = nullptr;
  // The gamma scaling factor.
  void const* gamma;
  // The beta term to add in GN.
  void const* beta;
  // The temporary buffer to do the global parallel reduction. Size:
  // BLOCKS_PER_BATCH x C x 2.
  float* redBuffer;

  float* var_data;

  // The number of instances in the batch.
  int32_t n;
  // The depth, height and width of each activation map.
  int32_t d, h, w;
  // The number of channels.
  int32_t c;
  // The number of groups.
  int32_t groups;
  // Do we apply the Silu activation function?
  bool withSilu;
  //
  bool y_same_with_x = false;
  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (d * h * w) and the number of
  // activations per block.
  int32_t dhw, dhwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock, cPerGroup;

  // The precomputed stride between instances.
  int32_t dhwc;
  // The inverse of dhwc in floats (to compute mean/var).
  float invDHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;
  // epsilon, Constant for numerical stability
  float eps;
  // for NCDHW32 int8 use
  float dqScaleIn;
  float inv_qScale;
};

template <typename T>
class groupNormNDHWCSum {
 public:
  void operator()(GroupNormNDHWCParams<T>* params, const gpuStream_t stream);
};

template <typename T>
class groupNormNDHWCScale {
 public:
  void operator()(const GroupNormNDHWCParams<T>& params,
                  const gpuStream_t stream);
};

}  // namespace phi
