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

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/trans_layernorm_op_plugin.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#ifdef TRT_PLUGIN_FP16_AVALIABLE
#define FINAL_MASK 0xffffffff

template <int UNROLL_FACTOR>
__global__ void generalAddBiasResidualLayerNormOpt2(
    half2 *normed_output,
    half2 *output,
    const half2 *__restrict src,
    const half2 *__restrict gamma,
    const half2 *__restrict beta,
    int m,
    int n,
    float epsilon) {
  __shared__ float s_mean;
  __shared__ float s_variance;
  float x_sum = 0.0f;
  float x2_sum = 0.0f;
  const int b_offset = blockIdx.x * n;

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = b_offset + i;
    float val_1 = 0.0f;
    float val_2 = 0.0f;
    half2 tmp;
    {
      tmp = __ldg(&src[index]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    tmp.x = __float2half_rn(val_1);
    tmp.y = __float2half_rn(val_2);
    output[index] = tmp;
    x_sum += val_1 + val_2;
    x2_sum += val_1 * val_1 + val_2 * val_2;
  }
  float sums[2];
  sums[0] = x_sum;
  sums[1] = x2_sum;
  phi::funcs::BlockReduceSumV2<float, 2>(sums);

  if (threadIdx.x == 0) {
    s_mean = sums[0] / n / 2;
    s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + epsilon);
  }
  __syncthreads();

  half2 mean_2 = __float2half2_rn(s_mean);
  half2 var_2 = __float2half2_rn(s_variance);

#pragma unroll UNROLL_FACTOR
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int index = b_offset + i;
    half2 val = __hmul2(__hmul2(__hsub2(output[index], mean_2), var_2),
                        __ldg(&gamma[i]));
    if (beta) {
      val = __hadd2(val, __ldg(&beta[i]));
    }
    normed_output[index] = val;
  }
}

#define HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(UNROLL_FACTOR)                \
  generalAddBiasResidualLayerNormOpt2<UNROLL_FACTOR>                         \
      <<<rows, block, 0, stream>>>(reinterpret_cast<half2 *>(layernorm_dst), \
                                   reinterpret_cast<half2 *>(dst),           \
                                   (const half2 *)input,                     \
                                   (const half2 *)fp16_scale_gpu_,           \
                                   (const half2 *)fp16_bias_gpu_,            \
                                   rows,                                     \
                                   half_n,                                   \
                                   eps);

#endif

using half = phi::dtype::float16;

int TransLayerNormPluginDynamic::initialize() TRT_NOEXCEPT {
  if (!with_fp16_) {
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
    cudaMemcpy(bias_gpu_,
               bias_.data(),
               bias_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
    cudaMemcpy(scale_gpu_,
               scale_.data(),
               scale_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    std::vector<half> fp16_bias_;
    std::vector<half> fp16_scale_;
    fp16_bias_.resize(bias_.size());
    fp16_scale_.resize(scale_.size());
    for (int i = 0; i < bias_.size(); i++) {
      fp16_bias_[i] = static_cast<half>(bias_[i]);
    }
    for (int i = 0; i < scale_.size(); i++) {
      fp16_scale_[i] = static_cast<half>(scale_[i]);
    }
    cudaMalloc(&fp16_bias_gpu_, sizeof(half) * fp16_bias_.size());
    cudaMemcpy(fp16_bias_gpu_,
               fp16_bias_.data(),
               fp16_bias_.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMalloc(&fp16_scale_gpu_, sizeof(half) * fp16_scale_.size());
    cudaMemcpy(fp16_scale_gpu_,
               fp16_scale_.data(),
               fp16_scale_.size() * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  return 0;
}

void TransLayerNormPluginDynamic::terminate() TRT_NOEXCEPT {
  if (bias_gpu_) {
    cudaFree(bias_gpu_);
    bias_gpu_ = nullptr;
  }
  if (scale_gpu_) {
    cudaFree(scale_gpu_);
    scale_gpu_ = nullptr;
  }
}

nvinfer1::DimsExprs TransLayerNormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputDims[0].d[0];
  ret.d[1] = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                    *inputDims[0].d[2],
                                    *inputDims[0].d[3]);
  ret.d[2] = inputDims[0].d[1];
  return ret;
}

bool TransLayerNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of layernorm plugin shoule not be nullptr."));
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
      return ((in.type == nvinfer1::DataType::kFLOAT ||
               in.type == nvinfer1::DataType::kHALF) &&
              (
                  // TODO(wangbojun) linear input support latter
                  //   in.format == nvinfer1::PluginFormat::kLINEAR ||
                  in.format == nvinfer1::PluginFormat::kHWC8));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  if (pos == 1) {
    if (with_fp16_) {
      return ((in.type == nvinfer1::DataType::kFLOAT ||
               in.type == nvinfer1::DataType::kHALF) &&
              (in.format == nvinfer1::PluginFormat::kLINEAR));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  if (pos == 2) {
    if (with_fp16_) {
      return ((in.type == nvinfer1::DataType::kFLOAT ||
               in.type == nvinfer1::DataType::kHALF) &&
              (in.format == nvinfer1::PluginFormat::kLINEAR));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
}

void TransLayerNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {
  const auto &input_dims = in[0].desc.dims;
  int statis_num = 1;
  for (int i = 0; i < begin_norm_axis_; i++) {
    statis_num *= input_dims.d[i];
  }
  mean_shape_[0] = statis_num;
  variance_shape_[0] = statis_num;
}

nvinfer1::DataType TransLayerNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_inputs,
                    1,
                    platform::errors::InvalidArgument(
                        "The LayerNormPlugin only has one input, so the "
                        "nb_inputs value should be 1, but get %d.",
                        nb_inputs));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}

int TransLayerNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  int begin_norm_axis = begin_norm_axis_;
  float eps = eps_;

  std::vector<int> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  // in dynamic shape
  // the batch num should be involved in mean/variance shape
  PADDLE_ENFORCE_EQ(1,
                    mean_shape_.size(),
                    platform::errors::InvalidArgument(
                        "Size of mean_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_EQ(1,
                    variance_shape_.size(),
                    platform::errors::InvalidArgument(
                        "Size of variance_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_GE(mean_shape_[0],
                    0,
                    platform::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        mean_shape_[0]));
  PADDLE_ENFORCE_GE(variance_shape_[0],
                    0,
                    platform::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        variance_shape_[0]));

  const auto input_ddim = phi::make_ddim(input_shape);
  int feature_size = static_cast<int>(input_ddim[1]);
  PADDLE_ENFORCE_EQ(feature_size,
                    scale_.size(),
                    platform::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size,
                        scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size,
                    bias_.size(),
                    platform::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size,
                        bias_.size()));

  int device_id;
  cudaGetDevice(&device_id);
  mean_t.Resize(phi::make_ddim(mean_shape_));
  variance_t.Resize(phi::make_ddim(variance_shape_));
  float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *variance_d =
      variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    if (input_desc[0].format == nvinfer1::PluginFormat::kLINEAR) {
      // TODO
    } else if (input_desc[0].format == nvinfer1::PluginFormat::kHWC8) {
      VLOG(1) << "TRT Plugin DataType selected. trans_layernorm-->fp32";
      const float *input = reinterpret_cast<const float *>(inputs[0]);
      float *output = static_cast<float *>(outputs[0]);
      phi::LayerNormDirectCUDAFunctor<float, float> layer_norm;
      layer_norm(stream,
                 input,
                 input_shape,
                 bias_gpu_,
                 scale_gpu_,
                 output,
                 mean_d,
                 variance_d,
                 begin_norm_axis,
                 eps);
    }
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. trans_layernorm-->fp16";

    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *layernorm_dst = static_cast<half *>(outputs[0]);
    half *dst = static_cast<half *>(outputs[1]);
    if (input_desc[0].format == nvinfer1::PluginFormat::kLINEAR) {
      // TODO
    } else if (input_desc[0].format == nvinfer1::PluginFormat::kHWC8) {
      // std::cout<<"@ trans_layernorm_khwc8 input shape:"<<std::endl;
      // for (int i = 0; i < input_dims.nbDims; i++) {
      //   std::cout<<input_shape[i]<<' ';
      // }
      // std::cout<<std::endl;
      // call
      int hidden = input_shape[1];
      const size_t rows =
          static_cast<size_t>(input_shape[0] * input_shape[2] *
                              input_shape[3]);  // batch * seq_length

      int half_n = hidden / 2;
      int half_n_32 = (half_n + 31) / 32 * 32;
      dim3 block(std::min(half_n_32, 512));
      int rolls_per_thread = half_n / block.x;
      int unroll_factor = 8;
      while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
        unroll_factor /= 2;
      }
      switch (unroll_factor) {
        case 1:
          HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(1);
          break;
        case 2:
          HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(2);
          break;
        case 4:
          HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(4);
          break;
        case 8:
          HALF2_ADD_BIAS_RESIDUAL_LAYERNORM_OPT2(8);
          break;
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "Invalid UNROLL_FACTOR in preln_residual_bias trt plugin."));
      }
    }
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("The TransLayerNormPluginDynamic TRT Plugin's "
                                "input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
