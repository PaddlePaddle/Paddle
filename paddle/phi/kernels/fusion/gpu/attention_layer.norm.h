// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"

namespace phi {
namespace fusion {

// NOTE: T must be the same as OutType in ComputeBackward
template <typename T, typename InType = T, typename OutType = T>
class AttnLayerNorm {
 public:
  AttnLayerNorm(const phi::GPUContext& dev_ctx,
                float epsilon,
                int64_t batch_size,
                int64_t feature_size)
      : dev_ctx_(dev_ctx),
        epsilon_(epsilon),
        batch_size_(batch_size),
        feature_size_(feature_size) {}

  ~AttnLayerNorm() {}

  void ComputeForward(const InType* x_data,
                      const phi::funcs::LayerNormParamType<T>* scale_data,
                      const phi::funcs::LayerNormParamType<T>* bias_data,
                      OutType* y_data,
                      phi::funcs::LayerNormParamType<T>* mean_data,
                      phi::funcs::LayerNormParamType<T>* var_data,
                      const float* dequant_out_scale_data = nullptr,
                      const int quant_out_scale_offset = 0,
                      const float quant_in_scale = 1.0,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    auto stream = dev_ctx_.stream();

    switch (phi::funcs::GetDesiredBlockDim(feature_size_)) {
      FIXED_BLOCK_DIM_CASE(
          phi::funcs::LayerNormForward<T,
                                       phi::funcs::LayerNormParamType<T>,
                                       kBlockDim,
                                       false,
                                       InType,
                                       OutType>
          <<<batch_size_, kBlockDim, 0, stream>>>(x_data,
                                                  scale_data,
                                                  bias_data,
                                                  y_data,
                                                  mean_data,
                                                  var_data,
                                                  epsilon_,
                                                  feature_size_,
                                                  dequant_out_scale_data,
                                                  quant_out_scale_offset,
                                                  quant_in_scale,
                                                  quant_round_type,
                                                  quant_max_bound,
                                                  quant_min_bound));
      default:
        PADDLE_THROW(common::errors::InvalidArgument(
            "Feature_size must be larger than 1"));
        break;
    }
  }

  void ComputeBackward(const T* x_data,
                       const T* d_y_data,
                       const phi::funcs::LayerNormParamType<T>* scale_data,
                       const phi::funcs::LayerNormParamType<T>* mean_data,
                       const phi::funcs::LayerNormParamType<T>* var_data,
                       T* d_x_data,
                       phi::funcs::LayerNormParamType<T>* d_scale_data,
                       phi::funcs::LayerNormParamType<T>* d_bias_data) {
    phi::funcs::LayerNormBackward<T, phi::funcs::LayerNormParamType<T>>(
        x_data,
        d_y_data,
        scale_data,
        mean_data,
        var_data,
        d_x_data,
        d_scale_data,
        d_bias_data,
        epsilon_,
        batch_size_,
        feature_size_,
        dev_ctx_);
  }

 private:
  const phi::GPUContext& dev_ctx_;

  int64_t batch_size_;
  int64_t feature_size_;

  float epsilon_;
};

}  // namespace fusion
}  // namespace phi
