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

#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/preln_residual_bias_plugin.h"
#include "paddle/fluid/operators/fused/fused_dropout_common.h"
#include "paddle/fluid/operators/fused/fused_layernorm_residual_dropout_bias.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
using half = phi::dtype::float16;

#if IS_TRT_VERSION_GE(6000)
int PrelnResidualBiasPluginDynamic::initialize() TRT_NOEXCEPT {
  cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
  cudaMemcpy(bias_gpu_,
             bias_.data(),
             bias_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
  cudaMemcpy(scale_gpu_,
             scale_.data(),
             scale_size_ * sizeof(float),
             cudaMemcpyHostToDevice);
  if (ele_bias_size_ > 0) {
    if (with_fp16_) {
      cudaMalloc(&ele_bias_gpu_, sizeof(half) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp16_ele_bias_.data(),
                 ele_bias_size_ * sizeof(half),
                 cudaMemcpyHostToDevice);
    } else {
      cudaMalloc(&ele_bias_gpu_, sizeof(float) * ele_bias_size_);
      cudaMemcpy(ele_bias_gpu_,
                 fp32_ele_bias_.data(),
                 ele_bias_size_ * sizeof(float),
                 cudaMemcpyHostToDevice);
    }
  } else {
    ele_bias_gpu_ = nullptr;
  }

  return 0;
}

void PrelnResidualBiasPluginDynamic::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
  if (ele_bias_gpu_) {
    cudaFree(ele_bias_gpu_);
    ele_bias_gpu_ = nullptr;
  }
}

nvinfer1::IPluginV2DynamicExt *PrelnResidualBiasPluginDynamic::clone() const
    TRT_NOEXCEPT {
  PrelnResidualBiasPluginDynamic *ptr = nullptr;
  if (with_fp16_) {
    ptr = new PrelnResidualBiasPluginDynamic(bias_.data(),
                                             scale_.data(),
                                             fp16_ele_bias_.data(),
                                             bias_size_,
                                             scale_size_,
                                             ele_bias_size_,
                                             eps_,
                                             with_fp16_);
  } else {
    ptr = new PrelnResidualBiasPluginDynamic(bias_.data(),
                                             scale_.data(),
                                             fp32_ele_bias_.data(),
                                             bias_size_,
                                             scale_size_,
                                             ele_bias_size_,
                                             eps_,
                                             with_fp16_);
  }

  ptr->bias_gpu_ = bias_gpu_;
  ptr->scale_gpu_ = scale_gpu_;
  ptr->ele_bias_gpu_ = ele_bias_gpu_;
  return ptr;
}

const char *PrelnResidualBiasPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "preln_residual_bias_plugin_dynamic";
}

int PrelnResidualBiasPluginDynamic::getNbOutputs() const TRT_NOEXCEPT {
  return 2;
}

size_t PrelnResidualBiasPluginDynamic::getSerializationSize() const
    TRT_NOEXCEPT {
  size_t ser_size = SerializedSize(bias_) + SerializedSize(scale_) +
                    SerializedSize(fp32_ele_bias_) +
                    SerializedSize(fp16_ele_bias_) +
                    SerializedSize(bias_size_) + SerializedSize(scale_size_) +
                    SerializedSize(ele_bias_size_) + SerializedSize(eps_) +
                    SerializedSize(with_fp16_);
  return ser_size;
}
void PrelnResidualBiasPluginDynamic::serialize(void *buffer) const
    TRT_NOEXCEPT {
  SerializeValue(&buffer, bias_);
  SerializeValue(&buffer, scale_);
  SerializeValue(&buffer, fp32_ele_bias_);
  SerializeValue(&buffer, fp16_ele_bias_);
  SerializeValue(&buffer, bias_size_);
  SerializeValue(&buffer, scale_size_);
  SerializeValue(&buffer, ele_bias_size_);
  SerializeValue(&buffer, eps_);
  SerializeValue(&buffer, with_fp16_);
}

nvinfer1::DimsExprs PrelnResidualBiasPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  if (output_index < 2) {
    return inputs[0];
  } else {  // moving mean and var
    nvinfer1::DimsExprs ret;
    ret.nbDims = 1;
    ret.d[0] = inputs[0].d[2];
    return ret;
  }
}

bool PrelnResidualBiasPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      return (in.type == nvinfer1::DataType::kHALF) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
      PADDLE_THROW(
          platform::errors::Fatal("TRT plugin supported FP16 is not available "
                                  "while with_fp16 is set true."));
#endif
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];

  if (pos == 1) {
    return in.type == prev.type && in.format == prev.format;
  }

  // output
  return in.type == prev.type && in.format == prev.format;
}

void PrelnResidualBiasPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nb_outputs) TRT_NOEXCEPT {}

size_t PrelnResidualBiasPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nb_outputs) const TRT_NOEXCEPT {
  return 0;
}

nvinfer1::DataType PrelnResidualBiasPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

void PrelnResidualBiasPluginDynamic::destroy() TRT_NOEXCEPT { delete this; }

int PrelnResidualBiasPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int hidden = input_dims.d[2];
  const size_t rows = static_cast<size_t>(
      input_dims.d[0] * input_dims.d[1]);  // batch * seq_length
  const size_t cols = static_cast<size_t>(input_dims.d[2]);

  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp32";
    const float *input1 = static_cast<const float *>(inputs[0]);
    const float *input2 = static_cast<const float *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const float *src = input2;
    const float *residual = input1;
    const float *bias = static_cast<float *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    float *dst = static_cast<float *>(outputs[1]);
    float *layernorm_dst = static_cast<float *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;
    paddle::operators::FusedLayernormResidualDropoutBiasFunctor<float,
                                                                uint8_t,
                                                                VecSize,
                                                                float,
                                                                false>()(
        rows,
        cols,
        seed,
        dropout_prob,
        is_upscale_in_train,
        is_test,
        increment,
        epsilon,
        src,
        residual,
        bias,
        scale,
        layernorm_bias,
        mask_data,
        dst,
        layernorm_dst,
        mean,
        var,
        stream);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. PrelnResidualBias-->fp16";
    const half *input1 = static_cast<const half *>(inputs[0]);
    const half *input2 = static_cast<const half *>(inputs[1]);

    uint64_t seed = 0;
    const float dropout_prob = 0.;
    const bool is_upscale_in_train = false;
    const bool is_test = true;
    const uint64_t increment = 0;
    const float epsilon = eps_;
    const half *src = input2;
    const half *residual = input1;
    const half *bias = static_cast<half *>(ele_bias_gpu_);
    const float *scale = scale_gpu_;
    const float *layernorm_bias = bias_gpu_;
    uint8_t *mask_data = nullptr;
    half *dst = static_cast<half *>(outputs[1]);
    half *layernorm_dst = static_cast<half *>(outputs[0]);
    float *mean = nullptr;
    float *var = nullptr;
    const int VecSize = 8;
    paddle::operators::FusedLayernormResidualDropoutBiasFunctor<half,
                                                                uint8_t,
                                                                VecSize,
                                                                float,
                                                                false>()(
        rows,
        cols,
        seed,
        dropout_prob,
        is_upscale_in_train,
        is_test,
        increment,
        epsilon,
        src,
        residual,
        bias,
        scale,
        layernorm_bias,
        mask_data,
        dst,
        layernorm_dst,
        mean,
        var,
        stream);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "The Ernie(Bert) tensorRT plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recomplie it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("The PrelnResidualBias TRT Plugin's input type "
                                "should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

const char *PrelnResidualBiasPluginDynamicCreator::getPluginName() const
    TRT_NOEXCEPT {
  return "preln_residual_bias_plugin_dynamic";
}

const char *PrelnResidualBiasPluginDynamicCreator::getPluginVersion() const
    TRT_NOEXCEPT {
  return "1";
}

nvinfer1::IPluginV2 *PrelnResidualBiasPluginDynamicCreator::deserializePlugin(
    const char *name,
    const void *serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  return new PrelnResidualBiasPluginDynamic(serial_data, serial_length);
}

#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
