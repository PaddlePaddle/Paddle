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

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/backends/dynload/cublasLt.h"
#endif

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/fusion/gpu/fused_bias_act_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_act_bias.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_common.h"
#include "paddle/phi/kernels/fusion/gpu/fused_layernorm_residual_dropout_bias.h"
#include "paddle/phi/kernels/fusion/gpu/fused_residual_dropout_bias.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"

COMMON_DECLARE_bool(use_fast_math);

namespace phi {
namespace fusion {

struct DropoutParam {
  uint64_t seed;
  float dropout_prob;
  bool is_upscale_in_train;
  bool is_test;
  bool fix_seed;
  int increment{};
  const phi::DenseTensor* tensor_seed;
  int seed_val;

  DropoutParam() {
    fix_seed = false;
    seed = 0;
    is_test = false;
    is_upscale_in_train = false;
    dropout_prob = 0.5;
    tensor_seed = nullptr;
    seed_val = 0;
  }

  DropoutParam(bool fix_seed_,
               uint64_t seed_,
               bool is_test_,
               bool is_upscale_in_train_,
               float dropout_prob_,
               const phi::DenseTensor* tensor_seed_,
               int seed_val_) {
    fix_seed = fix_seed_;
    seed = seed_;
    is_test = is_test_;
    is_upscale_in_train = is_upscale_in_train_;
    dropout_prob = dropout_prob_;
    tensor_seed = tensor_seed_;
    seed_val = seed_val_;
  }

  int UpdateSeedAndIncrement(const phi::GPUContext& dev_ctx, const int offset) {
    uint64_t tmp_increment;
    phi::funcs::GetSeedDataAndIncrement(dev_ctx,
                                        tensor_seed,
                                        fix_seed,
                                        seed_val,
                                        offset,
                                        &seed,
                                        &tmp_increment);
    increment = static_cast<int>(tmp_increment);
    return increment;
  }
};

template <typename T>
struct DataTypeTraits {
  using DataType = T;
};

template <>
struct DataTypeTraits<phi::dtype::float16> {
  // Since LayerNormDirectCUDAFunctor register half type, we need to convert
  // phi::float16 to half.
  using DataType = half;
};

template <typename T,
          typename MaskType,
          typename InType = T,
          typename OutType = T>
class FusedDropoutHelper {
 private:
  int GetIncrement(const phi::GPUContext& ctx) {
    const int VecSize = MAX_CACHE_BYTES / sizeof(T);
    const int real_vec_size = cols_ % VecSize == 0 ? VecSize : 1;
    auto config = Get1DBlocksAnd2DGrids(ctx,
                                        static_cast<uint64_t>(rows_),
                                        static_cast<uint64_t>(cols_),
                                        real_vec_size);
    int increment = ((cols_ - 1) / (config.thread_per_block.x *
                                    config.block_per_grid.x * real_vec_size) +
                     1) *
                    real_vec_size;
    increment = dropout_param_.UpdateSeedAndIncrement(ctx, increment);
    return increment;
  }

 public:
  FusedDropoutHelper() {}
  FusedDropoutHelper(const phi::GPUContext& ctx,
                     const int rows,
                     const int cols,
                     const DropoutParam& dropout_param,
                     const float residual_alpha = 1.0) {
    rows_ = rows;
    cols_ = cols;
    dropout_param_ = dropout_param;
    residual_alpha_ = residual_alpha;
  }

  // out = residual + dropout( src + bias )
  void ResidualDropoutBias(const phi::GPUContext& ctx,
                           const InType* src,
                           const T* residual,
                           const T* bias,
                           OutType* out,
                           MaskType* mask,
                           const float quant_last_in_scale = 1.0,
                           const float* dequant_out_scale_data = nullptr,
                           const float quant_next_in_scale = 1.0) {
    auto increment = GetIncrement(ctx);
    LaunchResidualDropoutBias<T, MaskType, InType, OutType>(
        rows_,
        cols_,
        increment,
        dropout_param_.seed,
        dropout_param_.dropout_prob,
        dropout_param_.is_test,
        dropout_param_.is_upscale_in_train,
        src,
        residual,
        bias,
        mask,
        out,
        ctx,
        quant_last_in_scale,
        dequant_out_scale_data,
        quant_next_in_scale,
        residual_alpha_);
  }

  void ResidualDropoutBiasGrad(const phi::GPUContext& ctx,
                               const T* d_out,
                               const MaskType* mask,
                               T* d_src,
                               T* d_residual,
                               T* d_bias) {
    LaunchResidualDropoutBiasGrad<T, uint8_t>(
        d_out,
        mask,
        dropout_param_.dropout_prob,
        dropout_param_.is_upscale_in_train,
        rows_,
        cols_,
        d_src,
        d_bias,
        ctx);
    if (d_residual) {
      phi::memory_utils::Copy(ctx.GetPlace(),
                              d_residual,
                              ctx.GetPlace(),
                              d_out,
                              rows_ * cols_ * sizeof(T),
                              ctx.stream());
    }
  }

  // out = dropout(activation(src + bias))
  void DropoutActBias(const phi::GPUContext& ctx,
                      const InType* src,
                      const T* bias,
                      const std::string& act_method,
                      OutType* out,
                      MaskType* mask,
                      const float quant_last_in_scale = 1.0,
                      const float* dequant_out_scale_data = nullptr,
                      const float quant_next_in_scale = 1.0,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    auto increment = GetIncrement(ctx);
    if (act_method == "gelu") {
      if (FLAGS_use_fast_math) {
        phi::fusion::FastGeluFunctor<T> fast_gelu;
        phi::fusion::LaunchDropoutActBias<T,
                                          MaskType,
                                          phi::fusion::FastGeluFunctor<T>,
                                          InType,
                                          OutType>(
            fast_gelu,
            dropout_param_.seed,
            rows_,
            cols_,
            dropout_param_.increment,
            dropout_param_.dropout_prob,
            dropout_param_.is_upscale_in_train,
            dropout_param_.is_test,
            src,
            bias,
            out,
            mask,
            ctx,
            quant_last_in_scale,
            dequant_out_scale_data,
            quant_next_in_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound);
      } else {
        phi::fusion::LayerNormParamTypeGeluFunctor<T> gelu;
        phi::fusion::LaunchDropoutActBias<
            T,
            MaskType,
            phi::fusion::LayerNormParamTypeGeluFunctor<T>,
            InType,
            OutType>(gelu,
                     dropout_param_.seed,
                     rows_,
                     cols_,
                     dropout_param_.increment,
                     dropout_param_.dropout_prob,
                     dropout_param_.is_upscale_in_train,
                     dropout_param_.is_test,
                     src,
                     bias,
                     out,
                     mask,
                     ctx,
                     quant_last_in_scale,
                     dequant_out_scale_data,
                     quant_next_in_scale,
                     quant_round_type,
                     quant_max_bound,
                     quant_min_bound);
      }
    } else if (act_method == "relu") {
      phi::funcs::ReluFunctor<T> relu;
      phi::fusion::LaunchDropoutActBias<T,
                                        MaskType,
                                        phi::funcs::ReluFunctor<T>,
                                        InType,
                                        OutType>(
          relu,
          dropout_param_.seed,
          rows_,
          cols_,
          increment,
          dropout_param_.dropout_prob,
          dropout_param_.is_upscale_in_train,
          dropout_param_.is_test,
          src,
          bias,
          out,
          mask,
          ctx,
          quant_last_in_scale,
          dequant_out_scale_data,
          quant_next_in_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Currently only supports gelu or relu activation functions!"));
    }
  }

  void DropoutActBiasGrad(const phi::GPUContext& ctx,
                          const T* dout,
                          const T* src,
                          const T* bias,
                          const MaskType* mask,
                          T* d_src,
                          T* d_bias,
                          const std::string& act_method) {
    if (act_method == "gelu") {
      phi::fusion::GeluGradFunctor<T> gelu_grad;
      phi::fusion::LaunchDropoutActBiasGrad<T,
                                            MaskType,
                                            phi::fusion::GeluGradFunctor<T>>(
          gelu_grad,
          dout,
          mask,
          src,
          bias,
          dropout_param_.dropout_prob,
          dropout_param_.is_upscale_in_train,
          rows_,
          cols_,
          d_src,
          d_bias,
          ctx);
    } else if (act_method == "relu") {
      phi::funcs::ReluGradFunctor<T> relu_grad;
      phi::fusion::
          LaunchDropoutActBiasGrad<T, MaskType, phi::funcs::ReluGradFunctor<T>>(
              relu_grad,
              dout,
              mask,
              src,
              bias,
              dropout_param_.dropout_prob,
              dropout_param_.is_upscale_in_train,
              rows_,
              cols_,
              d_src,
              d_bias,
              ctx);
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Currently only supports gelu or relu activation functions!"));
    }
  }

 protected:
  int rows_;
  int cols_;
  DropoutParam dropout_param_;
  float residual_alpha_;
};

template <typename T,
          typename MaskType,
          typename InType = T,
          typename OutType = T>
class FusedDropoutLayerNormHelper
    : public FusedDropoutHelper<T, MaskType, InType, OutType> {
 public:
  FusedDropoutLayerNormHelper() {}
  FusedDropoutLayerNormHelper(const int rows,
                              const int cols,
                              const float epsilon,
                              const float residual_alpha = 1.0) {
    using U = phi::funcs::LayerNormParamType<T>;
    this->rows_ = rows;
    this->cols_ = cols;
    epsilon_ = epsilon;
    this->residual_alpha_ = residual_alpha;
  }

  FusedDropoutLayerNormHelper(const phi::GPUContext& ctx,
                              const int rows,
                              const int cols,
                              const DropoutParam& dropout_param,
                              const float epsilon,
                              const float residual_alpha = 1.0)
      : FusedDropoutHelper<T, MaskType, InType, OutType>(
            ctx, rows, cols, dropout_param, residual_alpha) {
    using U = phi::funcs::LayerNormParamType<T>;
    epsilon_ = epsilon;
  }

  // call layer_norm
  void LayerNorm(const phi::GPUContext& ctx,
                 const InType* src,
                 const phi::funcs::LayerNormParamType<T>* gamma,
                 const phi::funcs::LayerNormParamType<T>* beta,
                 OutType* out,
                 phi::funcs::LayerNormParamType<T>* mean,
                 phi::funcs::LayerNormParamType<T>* variance) {
    using InDataType = typename DataTypeTraits<InType>::DataType;
    using OutDataType = typename DataTypeTraits<OutType>::DataType;

    phi::LayerNormDirectCUDAFunctor<InDataType,
                                    phi::funcs::LayerNormParamType<T>>
        layer_norm;
    std::vector<int> src_shape{this->rows_, this->cols_};
    layer_norm(ctx.stream(),
               reinterpret_cast<const InDataType*>(src),
               src_shape,
               beta,
               gamma,
               reinterpret_cast<OutDataType*>(out),
               mean,
               variance,
               1,
               epsilon_);
  }

  void LayerNormGrad(const phi::GPUContext& ctx,
                     const T* dout,
                     const T* src,
                     const phi::funcs::LayerNormParamType<T>* gamma,
                     const phi::funcs::LayerNormParamType<T>* mean,
                     const phi::funcs::LayerNormParamType<T>* variance,
                     T* d_src,
                     phi::funcs::LayerNormParamType<T>* d_scale,
                     phi::funcs::LayerNormParamType<T>* d_bias) {
    using U = phi::funcs::LayerNormParamType<T>;
    phi::funcs::LayerNormBackward<T, U>(src,
                                        dout,
                                        gamma,
                                        mean,
                                        variance,
                                        d_src,
                                        d_scale,
                                        d_bias,
                                        epsilon_,
                                        this->rows_,
                                        this->cols_,
                                        ctx);
  }

  // out = layernorm(residual + dropout(src + bias))
  template <typename P = phi::funcs::LayerNormParamType<T>,
            bool is_same_type = false>
  void LayernormResidualDropoutBias(
      const phi::GPUContext& ctx,
      const InType* src,
      const T* residual,
      const T* bias,
      const P* gamma,
      const P* beta,
      T* dropout_out,
      MaskType* mask,
      OutType* out,
      phi::funcs::LayerNormParamType<T>* mean,
      phi::funcs::LayerNormParamType<T>* variance,
      const float quant_last_in_scale = 1.0,
      const float* dequant_out_scale_data = nullptr,
      const float quant_next_in_scale = 1.0,
      const int quant_round_type = 1,
      const float quant_max_bound = 127.0,
      const float quant_min_bound = -127.0) {
    using U = phi::funcs::LayerNormParamType<T>;
    int vec_size = MAX_CACHE_BYTES / sizeof(T);
    if (this->cols_ % vec_size != 0) {
      vec_size = 1;
    }
    int threads = phi::funcs::GetDesiredBlockDim(this->cols_ / vec_size);
    int increment = ((this->cols_ - 1) / (threads * vec_size) + 1) * vec_size;
    increment = this->dropout_param_.UpdateSeedAndIncrement(ctx, increment);
    LaunchLayernormResidualDropoutBias<T,
                                       MaskType,
                                       U,
                                       is_same_type,
                                       InType,
                                       OutType>(
        this->rows_,
        this->cols_,
        increment,
        this->dropout_param_.seed,
        this->dropout_param_.dropout_prob,
        epsilon_,
        this->dropout_param_.is_upscale_in_train,
        this->dropout_param_.is_test,
        src,
        residual,
        bias,
        gamma,
        beta,
        mask,
        dropout_out,
        out,
        mean,
        variance,
        ctx,
        quant_last_in_scale,
        dequant_out_scale_data,
        quant_next_in_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        this->residual_alpha_);
  }

  template <typename P = phi::funcs::LayerNormParamType<T>,
            bool is_same_type = false>
  void LayernormResidualDropoutBiasGrad(
      const phi::GPUContext& ctx,
      const T* d_out,
      const T* layernorm_src,
      const MaskType* mask,
      const P* gamma,
      const phi::funcs::LayerNormParamType<T>* mean,
      const phi::funcs::LayerNormParamType<T>* variance,
      T* d_layernorm_src,
      P* d_scale,
      P* d_layernorm_bias,
      T* d_dropout_src,
      T* d_bias,
      T* d_residual) {
    using U = phi::funcs::LayerNormParamType<T>;
    bool can_call_1024_kernel = false;
    // Fast impl for cases when cols is 1024 and linear_bias is nullptr.
    // In fact, linear_bias is not nullptr is also feasible for impl.
    // Here, we do not support it.
    if (this->cols_ == 1024 && d_bias == nullptr && d_scale != nullptr &&
        d_layernorm_bias != nullptr && sizeof(T) <= 4) {
      can_call_1024_kernel = true;
    }
    VLOG(6) << "LaunchLayernormResidualDropoutGrad = " << can_call_1024_kernel;

    if (can_call_1024_kernel) {
      LaunchLayernormResidualDropoutGrad<T, U, MaskType, is_same_type>(
          ctx,
          this->rows_,
          this->cols_,
          epsilon_,
          this->dropout_param_.dropout_prob,
          this->dropout_param_.is_upscale_in_train,
          d_out,
          layernorm_src,
          gamma,
          mean,
          variance,
          mask,
          d_scale,
          d_layernorm_bias,
          d_residual,
          d_dropout_src);
    } else {
      phi::funcs::LayerNormBackward<T, U, is_same_type>(layernorm_src,
                                                        d_out,
                                                        gamma,
                                                        mean,
                                                        variance,
                                                        d_layernorm_src,
                                                        d_scale,
                                                        d_layernorm_bias,
                                                        epsilon_,
                                                        this->rows_,
                                                        this->cols_,
                                                        ctx);
      this->ResidualDropoutBiasGrad(
          ctx, d_layernorm_src, mask, d_dropout_src, d_residual, d_bias);
    }
  }

 protected:
  float epsilon_;
};

}  // namespace fusion
}  // namespace phi
