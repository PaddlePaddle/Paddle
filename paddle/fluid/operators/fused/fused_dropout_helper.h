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

#pragma once

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/fused/fused_dropout_act_bias.h"
#include "paddle/fluid/operators/fused/fused_layernorm_residual_dropout_bias.h"
#include "paddle/fluid/operators/fused/fused_residual_dropout_bias.h"
#include "paddle/fluid/operators/math/functors.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

struct DropoutParam {
  uint64_t seed;
  float dropout_prob;
  bool is_upscale_in_train;
  bool is_test;
  bool fix_seed;
  int increment;
  bool has_increment;

  DropoutParam() {
    fix_seed = false;
    seed = 0;
    is_test = false;
    is_upscale_in_train = false;
    has_increment = false;
    dropout_prob = 0.5;
  }

  /**
   * dropout_index: the index of dropout, such as FFN has two dropout,
   * so the dropout_index will 1 or 2.
   * the dropout param will defined as param1 or param2
   */
  DropoutParam(const framework::ExecutionContext& context,
               const int dropout_index) {
    std::string str_index = std::to_string(dropout_index);
    if (dropout_index == 0) {
      str_index = "";
    }
    dropout_prob = context.Attr<float>("dropout_prob" + str_index);
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation" + str_index);
    is_upscale_in_train = (dropout_implementation == "upscale_in_train");
    is_test = context.Attr<bool>("is_test" + str_index);
    fix_seed = context.Attr<bool>("fix_seed" + str_index);
    has_increment = false;

    std::string str_seed = "Seed" + str_index;
    auto* tensor_seed =
        context.HasInput(str_seed) ? context.Input<Tensor>(str_seed) : nullptr;
    int device_id =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()).GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    if (tensor_seed && platform::is_gpu_place(tensor_seed->place())) {
      framework::Tensor seed_cpu_tensor;
      TensorCopySync(*tensor_seed, platform::CPUPlace(), &seed_cpu_tensor);
      seed = static_cast<uint64_t>(seed_cpu_tensor.data<int>()[0]);
    } else if (gen_cuda->GetIsInitPy() && !fix_seed) {
      has_increment = true;
    } else {
      if (tensor_seed) {
        seed = *(tensor_seed->data<int>());
      } else {
        std::random_device rnd;
        seed = fix_seed ? context.Attr<int>("seed" + str_index) : rnd();
      }
    }
  }
  int UpdateSeedAndIncrement(const platform::CUDADeviceContext& ctx,
                             const int offset) {
    int device_id =
        BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace()).GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    auto seed_offset = gen_cuda->IncrementOffset(offset);
    seed = seed_offset.first;
    increment = static_cast<int>(seed_offset.second);
    return increment;
  }
};

template <typename T, typename MaskType>
class FusedDropoutHelper {
 private:
  int GetIncrement(const platform::CUDADeviceContext& ctx) {
    const int VecSize = MAX_CACHE_BYTES / sizeof(T);
    const int real_vec_size = cols_ % VecSize == 0 ? VecSize : 1;
    auto config =
        Get1DBlocksAnd2DGrids(ctx, static_cast<uint64_t>(rows_),
                              static_cast<uint64_t>(cols_), real_vec_size);
    int increment = ((cols_ - 1) / (config.thread_per_block.x *
                                    config.block_per_grid.x * real_vec_size) +
                     1) *
                    real_vec_size;
    if (dropout_param_.has_increment) {
      increment = dropout_param_.UpdateSeedAndIncrement(ctx, increment);
    }
    return increment;
  }

 public:
  FusedDropoutHelper() {}
  FusedDropoutHelper(const platform::CUDADeviceContext& ctx, const int rows,
                     const int cols, const DropoutParam& dropout_param) {
    rows_ = rows;
    cols_ = cols;
    dropout_param_ = dropout_param;
  }

  // out = residual + dropout( src + bias )
  void ResidualDropoutBias(const platform::CUDADeviceContext& ctx, const T* src,
                           const T* residual, const T* bias, T* out,
                           MaskType* mask) {
    auto increment = GetIncrement(ctx);
    LaunchResidualDropoutBias<T, MaskType>(
        rows_, cols_, increment, dropout_param_.seed,
        dropout_param_.dropout_prob, dropout_param_.is_test,
        dropout_param_.is_upscale_in_train, src, residual, bias, mask, out,
        ctx);
  }

  void ResidualDropoutBiasGrad(const platform::CUDADeviceContext& ctx,
                               const T* d_out, const MaskType* mask, T* d_src,
                               T* d_residual, T* d_bias) {
    LaunchResidualDropoutBiasGrad<T, uint8_t>(
        d_out, mask, dropout_param_.dropout_prob,
        dropout_param_.is_upscale_in_train, rows_, cols_, d_src, d_bias, ctx);
    auto cuda_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    memory::Copy(cuda_place, d_residual, cuda_place, d_out,
                 rows_ * cols_ * sizeof(T), ctx.stream());
  }

  // out = dropout(activation(src + bias))
  void DropoutActBias(const platform::CUDADeviceContext& ctx, const T* src,
                      const T* bias, const std::string& act_method, T* out,
                      MaskType* mask) {
    auto increment = GetIncrement(ctx);
    if (act_method == "gelu") {
      GeluFunctor<T> gelu;
      LaunchDropoutActBias<T, MaskType, GeluFunctor<T>>(
          gelu, dropout_param_.seed, rows_, cols_, dropout_param_.increment,
          dropout_param_.dropout_prob, dropout_param_.is_upscale_in_train,
          dropout_param_.is_test, src, bias, out, mask, ctx);
    } else if (act_method == "relu") {
      math::ReluFunctor<T> relu;
      LaunchDropoutActBias<T, MaskType, math::ReluFunctor<T>>(
          relu, dropout_param_.seed, rows_, cols_, increment,
          dropout_param_.dropout_prob, dropout_param_.is_upscale_in_train,
          dropout_param_.is_test, src, bias, out, mask, ctx);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "the activation only support gelu or relu!"));
    }
  }

  void DropoutActBiasGrad(const platform::CUDADeviceContext& ctx, const T* dout,
                          const T* src, const T* bias, const MaskType* mask,
                          T* d_src, T* d_bias, const std::string& act_method) {
    if (act_method == "gelu") {
      GeluGradFunctor<T> gelu_grad;
      LaunchDropoutActBiasGrad<T, MaskType, GeluGradFunctor<T>>(
          gelu_grad, dout, mask, src, bias, dropout_param_.dropout_prob,
          dropout_param_.is_upscale_in_train, rows_, cols_, d_src, d_bias, ctx);
    } else if (act_method == "relu") {
      math::ReluGradFunctor<T> relu_grad;
      LaunchDropoutActBiasGrad<T, MaskType, math::ReluGradFunctor<T>>(
          relu_grad, dout, mask, src, bias, dropout_param_.dropout_prob,
          dropout_param_.is_upscale_in_train, rows_, cols_, d_src, d_bias, ctx);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "the activation only support gelu or relu!"));
    }
  }

 protected:
  int rows_;
  int cols_;
  DropoutParam dropout_param_;
};

template <typename T, typename MaskType>
class FusedDropoutLayerNormHelper : public FusedDropoutHelper<T, MaskType> {
 public:
  FusedDropoutLayerNormHelper() {}
  FusedDropoutLayerNormHelper(const int rows, const int cols,
                              const float epsilon) {
    using U = LayerNormParamType<T>;
    this->rows_ = rows;
    this->cols_ = cols;
    epsilon_ = epsilon;
  }

  FusedDropoutLayerNormHelper(const platform::CUDADeviceContext& ctx,
                              const int rows, const int cols,
                              const DropoutParam& dropout_param,
                              const float epsilon)
      : FusedDropoutHelper<T, MaskType>(ctx, rows, cols, dropout_param) {
    using U = LayerNormParamType<T>;
    epsilon_ = epsilon;
  }

  // call layer_norm
  void LayerNorm(const platform::CUDADeviceContext& ctx, const T* src,
                 const LayerNormParamType<T>* gamma,
                 const LayerNormParamType<T>* beta, T* out,
                 LayerNormParamType<T>* mean, LayerNormParamType<T>* variance) {
    using U = LayerNormParamType<T>;
    switch (GetDesiredBlockDim(this->cols_)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<
              T, U, kBlockDim><<<this->rows_, kBlockDim, 0, ctx.stream()>>>(
              src, gamma, beta, out, mean, variance, epsilon_, this->cols_));
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Product from begin_norm_axis to end must be larger than 1"));
        break;
    }
  }

  void LayerNormGrad(const platform::CUDADeviceContext& ctx, const T* dout,
                     const T* src, const LayerNormParamType<T>* gamma,
                     const LayerNormParamType<T>* mean,
                     const LayerNormParamType<T>* variance, T* d_src,
                     LayerNormParamType<T>* d_scale,
                     LayerNormParamType<T>* d_bias) {
    using U = LayerNormParamType<T>;
    LayerNormBackward<T, U>(src, dout, gamma, mean, variance, d_src, d_scale,
                            d_bias, epsilon_, this->rows_, this->cols_, ctx);
  }

  // out = layernorm(residual + dropout(src + bias))
  void LayernormResidualDropoutBias(
      const platform::CUDADeviceContext& ctx, const T* src, const T* residual,
      const T* bias, const LayerNormParamType<T>* gamma,
      const LayerNormParamType<T>* beta, T* dropout_out, MaskType* mask, T* out,
      LayerNormParamType<T>* mean, LayerNormParamType<T>* variance) {
    using U = LayerNormParamType<T>;
    int vec_size = MAX_CACHE_BYTES / sizeof(T);
    if (this->cols_ % vec_size != 0) {
      vec_size = 1;
    }
    int threads = GetDesiredBlockDim(this->cols_ / vec_size);

    int increment = ((this->cols_ - 1) / (threads * vec_size) + 1) * vec_size;
    if (this->dropout_param_.has_increment) {
      increment = this->dropout_param_.UpdateSeedAndIncrement(ctx, increment);
    }

    LaunchLayernormResidualDropoutBias<T, MaskType>(
        this->rows_, this->cols_, increment, this->dropout_param_.seed,
        this->dropout_param_.dropout_prob, epsilon_,
        this->dropout_param_.is_upscale_in_train, this->dropout_param_.is_test,
        src, residual, bias, gamma, beta, mask, dropout_out, out, mean,
        variance, ctx);
  }

  void LayernormResidualDropoutBiasGrad(
      const platform::CUDADeviceContext& ctx, const T* d_out,
      const T* layernorm_src, const MaskType* mask,
      const LayerNormParamType<T>* gamma, const LayerNormParamType<T>* mean,
      const LayerNormParamType<T>* variance, T* d_layernorm_src,
      LayerNormParamType<T>* d_scale, LayerNormParamType<T>* d_layernorm_bias,
      T* d_dropout_src, T* d_bias, T* d_residual) {
    using U = LayerNormParamType<T>;
    LayerNormBackward<T, U>(layernorm_src, d_out, gamma, mean, variance,
                            d_layernorm_src, d_scale, d_layernorm_bias,
                            epsilon_, this->rows_, this->cols_, ctx);
    this->ResidualDropoutBiasGrad(ctx, d_layernorm_src, mask, d_dropout_src,
                                  d_residual, d_bias);
  }

 protected:
  float epsilon_;
};

}  // namespace operators
}  // namespace paddle
