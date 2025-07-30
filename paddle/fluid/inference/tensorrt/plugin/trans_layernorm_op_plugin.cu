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

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/trans_layernorm_op_plugin.h"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"
#include "paddle/phi/kernels/layer_norm_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

inline int getSMVersion() {
  const int device = phi::backends::gpu::GetCurrentDeviceId();
  const phi::gpuDeviceProp prop =
      phi::backends::gpu::GetDeviceProperties(device);
  return prop.major * 10 + prop.minor;
}

#ifdef TRT_PLUGIN_FP16_AVAILABLE
#define FINAL_MASK 0xffffffff

template <int UNROLL_FACTOR>
__global__ void GeneralResidualLayerNormOpt2(half2 *normed_output,
                                             half2 *output,
                                             const half2 *__restrict src,
                                             const half2 *__restrict gamma,
                                             const half2 *__restrict beta,
                                             int m,
                                             int n,
                                             float epsilon) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
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
    tmp = __ldg(&src[index]);
    val_1 += static_cast<float>(tmp.x);
    val_2 += static_cast<float>(tmp.y);
    output[index] = tmp;
    x_sum += val_1 + val_2;
    x2_sum += val_1 * val_1 + val_2 * val_2;
  }
  float sums[2];
  sums[0] = x_sum;
  sums[1] = x2_sum;
  phi::funcs::BlockReduceSumV2<float, 2>(sums);
  constexpr int Half2VecSize = 2;
  if (threadIdx.x == 0) {
    s_mean = sums[0] / n / Half2VecSize;
    s_variance = rsqrtf(sums[1] / n / Half2VecSize - s_mean * s_mean + epsilon);
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
#endif
}

#define HALF2_RESIDUAL_LAYERNORM_OPT2(UNROLL_FACTOR)                         \
  GeneralResidualLayerNormOpt2<UNROLL_FACTOR>                                \
      <<<rows, block, 0, stream>>>(reinterpret_cast<half2 *>(layernorm_dst), \
                                   reinterpret_cast<half2 *>(dst),           \
                                   (const half2 *)input,                     \
                                   (const half2 *)fp16_scale_gpu_,           \
                                   (const half2 *)fp16_bias_gpu_,            \
                                   rows,                                     \
                                   half_n,                                   \
                                   eps);

#endif

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
  if (fp16_bias_gpu_) {
    cudaFree(fp16_bias_gpu_);
    fp16_bias_gpu_ = nullptr;
  }
  if (fp16_scale_gpu_) {
    cudaFree(fp16_scale_gpu_);
    fp16_scale_gpu_ = nullptr;
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
  int feature_size = bias_.size();
  PADDLE_ENFORCE_GE(
      feature_size,
      0,
      common::errors::InvalidArgument(
          "The feature size of layernorm feature_size must be positive,"
          "but got:%d",
          feature_size));

  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      common::errors::InvalidArgument(
          "The input of layernorm plugin should not be nullptr."));
  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    if (with_fp16_) {
      if (feature_size % 8 == 0) {
        // now, we only support khwc8 for feature_size % 8 == 0
        return ((in.type == nvinfer1::DataType::kHALF) &&
                (in.format == nvinfer1::PluginFormat::kLINEAR ||
                 in.format == nvinfer1::PluginFormat::kHWC8));
      } else {
        return ((in.type == nvinfer1::DataType::kHALF) &&
                (in.format == nvinfer1::PluginFormat::kLINEAR));
      }
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  if (pos == 1) {
    if (with_fp16_) {
      return (in.type == in_out[0].type &&
              (in.format == nvinfer1::PluginFormat::kLINEAR));
    } else {
      return (in.type == in_out[0].type) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  if (pos == 2) {
    if (with_fp16_) {
      return (in.type == in_out[0].type &&
              (in.format == nvinfer1::PluginFormat::kLINEAR));
    } else {
      return (in.type == in_out[0].type) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
}

void TransLayerNormPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int nbOutputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      begin_norm_axis_,
      3,
      common::errors::InvalidArgument(
          "The transpose_LayerNorm Plugin only has begin_norm_axis_ = 3"
          "but get %d.",
          begin_norm_axis_));
  const auto &input_dims = in[0].desc.dims;
  int statis_num = input_dims.d[0] * input_dims.d[2] * input_dims.d[3];
  mean_shape_[0] = statis_num;
  variance_shape_[0] = statis_num;
}

nvinfer1::DataType TransLayerNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs,
      1,
      common::errors::InvalidArgument(
          "The transpose_LayerNorm Plugin only has one input, so the "
          "nb_inputs value should be 1, but get %d.",
          nb_inputs));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    common::errors::InvalidArgument(
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

  std::vector<int64_t> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  int input_numel = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_numel *= input_dims.d[i];
  }
  PADDLE_ENFORCE_EQ(1,
                    mean_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of mean_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_EQ(1,
                    variance_shape_.size(),
                    common::errors::InvalidArgument(
                        "Size of variance_shape vector should be equal to 1,"
                        "but got Size of mean_shape vector:%d",
                        mean_shape_.size()));
  PADDLE_ENFORCE_GE(mean_shape_[0],
                    0,
                    common::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        mean_shape_[0]));
  PADDLE_ENFORCE_GE(variance_shape_[0],
                    0,
                    common::errors::InvalidArgument(
                        "The size of mean vector should be positive,"
                        "but got:%d",
                        variance_shape_[0]));

  // transpose do not change numel
  int trans_result_numel = input_numel;
  std::vector<int64_t> trans_result_shape{
      input_shape[0], input_shape[2], input_shape[3], input_shape[1]};

  const auto input_ddim = common::make_ddim(input_shape);
  int feature_size = static_cast<int>(input_ddim[1]);
  PADDLE_ENFORCE_EQ(feature_size,
                    scale_.size(),
                    common::errors::InvalidArgument(
                        "scale's size should be equal to the feature_size,"
                        "but got feature_size:%d, scale's size:%d.",
                        feature_size,
                        scale_.size()));
  PADDLE_ENFORCE_EQ(feature_size,
                    bias_.size(),
                    common::errors::InvalidArgument(
                        "bias's size should be equal to the feature_size,"
                        "but got feature_size:%d, bias's size:%d.",
                        feature_size,
                        bias_.size()));

  int device_id = -1;
  cudaGetDevice(&device_id);
  PADDLE_ENFORCE_GE(
      device_id,
      0,
      common::errors::InvalidArgument("device_id should be positive,"
                                      "but got:%d",
                                      device_id));

  auto input_type = input_desc[0].type;

  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  auto *device_context = static_cast<phi::GPUContext *>(pool.Get(place));
  const phi::GPUContext &dev_ctx = *device_context;

  mean_t.Resize(common::make_ddim(mean_shape_));
  variance_t.Resize(common::make_ddim(variance_shape_));
  float *mean_d =
      dev_ctx.template Alloc<float>(&mean_t, mean_shape_[0] * sizeof(float));
  float *variance_d = dev_ctx.template Alloc<float>(
      &variance_t, variance_shape_[0] * sizeof(float));

  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. trans_layernorm-->fp32";
    VLOG(1) << "TRT Plugin format selected. trans_layernorm-->kLINEAR";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *layernorm_dst = static_cast<float *>(outputs[0]);
    float *dst = static_cast<float *>(outputs[1]);
    // transpose and norm do not change numel
    int trans_result_numel = input_numel;
    int norm_result_numel = input_numel;
    phi::DenseTensorMeta input_meta(phi::DataType::FLOAT32,
                                    common::make_ddim(input_shape));
    phi::DenseTensorMeta bias_meta(phi::DataType::FLOAT32,
                                   common::make_ddim({feature_size}));
    phi::DenseTensorMeta scale_meta(phi::DataType::FLOAT32,
                                    common::make_ddim({feature_size}));
    phi::DenseTensorMeta trans_result_meta(
        phi::DataType::FLOAT32, common::make_ddim(trans_result_shape));
    phi::DenseTensorMeta norm_result_meta(
        phi::DataType::FLOAT32, common::make_ddim(trans_result_shape));
    std::shared_ptr<phi::Allocation> input_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(input)),  // NOLINT
        input_numel * sizeof(float),
        place));
    std::shared_ptr<phi::Allocation> bias_alloc(
        new phi::Allocation(static_cast<float *>(bias_gpu_),  // NOLINT
                            feature_size * sizeof(float),
                            place));
    std::shared_ptr<phi::Allocation> scale_alloc(new phi::Allocation(
        static_cast<float *>(scale_gpu_), feature_size * sizeof(float), place));
    std::shared_ptr<phi::Allocation> trans_result_alloc(
        new phi::Allocation(static_cast<float *>(dst),  // NOLINT
                            trans_result_numel * sizeof(float),
                            place));
    std::shared_ptr<phi::Allocation> norm_result_alloc(
        new phi::Allocation(static_cast<float *>(layernorm_dst),  // NOLINT
                            norm_result_numel * sizeof(float),
                            place));

    const phi::DenseTensor input_tensor =
        phi::DenseTensor(input_alloc, input_meta);
    phi::DenseTensor bias_tensor = phi::DenseTensor(bias_alloc, bias_meta);
    phi::DenseTensor scale_tensor = phi::DenseTensor(scale_alloc, scale_meta);
    phi::DenseTensor trans_result_tensor =
        phi::DenseTensor(trans_result_alloc, trans_result_meta);
    phi::DenseTensor norm_result_tensor =
        phi::DenseTensor(norm_result_alloc, norm_result_meta);

    phi::TransposeKernel<float, phi::GPUContext>(dev_ctx,
                                                 input_tensor,
                                                 std::vector<int>{0, 2, 3, 1},
                                                 &trans_result_tensor);
    phi::LayerNormKernel<float, phi::GPUContext>(dev_ctx,
                                                 trans_result_tensor,
                                                 scale_tensor,
                                                 bias_tensor,
                                                 eps,
                                                 begin_norm_axis,
                                                 &norm_result_tensor,
                                                 &mean_t,
                                                 &variance_t);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. trans_layernorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *layernorm_dst = static_cast<half *>(outputs[0]);
    half *dst = static_cast<half *>(outputs[1]);
    if (input_desc[0].format == nvinfer1::PluginFormat::kLINEAR) {
      VLOG(1) << "TRT Plugin format selected. trans_layernorm-->kLINEAR";
      phi::DenseTensorMeta input_meta(phi::DataType::FLOAT16,
                                      common::make_ddim(input_shape));
      std::shared_ptr<phi::Allocation> input_alloc(new phi::Allocation(
          static_cast<void *>(const_cast<half *>(input)),  // NOLINT
          input_numel * sizeof(half),
          place));
      phi::DenseTensorMeta trans_result_meta(
          phi::DataType::FLOAT16, common::make_ddim(trans_result_shape));
      std::shared_ptr<phi::Allocation> trans_result_alloc(
          new phi::Allocation(static_cast<void *>(dst),  // NOLINT
                              trans_result_numel * sizeof(half),
                              place));
      const phi::DenseTensor input_tensor =
          phi::DenseTensor(input_alloc, input_meta);
      phi::DenseTensor trans_result_tensor =
          phi::DenseTensor(trans_result_alloc, trans_result_meta);
      phi::TransposeKernel<phi::dtype::float16, phi::GPUContext>(
          dev_ctx,
          input_tensor,
          std::vector<int>{0, 2, 3, 1},
          &trans_result_tensor);
      phi::LayerNormDirectCUDAFunctor<half, float> layer_norm;
      layer_norm(stream,
                 dst,
                 trans_result_shape,
                 bias_gpu_,
                 scale_gpu_,
                 layernorm_dst,
                 mean_d,
                 variance_d,
                 begin_norm_axis,
                 eps);
    } else if (input_desc[0].format == nvinfer1::PluginFormat::kHWC8) {
      VLOG(1) << "TRT Plugin format selected. trans_layernorm-->kHWC8";
      int sm = getSMVersion();
      // sm >= 60 to support __ldg
      if (sm >= 60) {
        int64_t hidden = input_shape[1];
        if (hidden % 2 == 0) {
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
              HALF2_RESIDUAL_LAYERNORM_OPT2(1);
              break;
            case 2:
              HALF2_RESIDUAL_LAYERNORM_OPT2(2);
              break;
            case 4:
              HALF2_RESIDUAL_LAYERNORM_OPT2(4);
              break;
            case 8:
              HALF2_RESIDUAL_LAYERNORM_OPT2(8);
              break;
            default:
              PADDLE_THROW(common::errors::Fatal(
                  "Invalid UNROLL_FACTOR in transpose_layernorm trt plugin."));
          }
        } else {
          cudaMemcpyAsync(
              dst, input, input_numel * sizeof(half), cudaMemcpyDeviceToDevice);
          phi::LayerNormDirectCUDAFunctor<half, float> layer_norm;
          layer_norm(stream,
                     input,
                     trans_result_shape,
                     bias_gpu_,
                     scale_gpu_,
                     layernorm_dst,
                     mean_d,
                     variance_d,
                     begin_norm_axis,
                     eps);
        }
      } else {
        cudaMemcpyAsync(
            dst, input, input_numel * sizeof(half), cudaMemcpyDeviceToDevice);
        phi::LayerNormDirectCUDAFunctor<half, float> layer_norm;
        layer_norm(stream,
                   input,
                   trans_result_shape,
                   bias_gpu_,
                   scale_gpu_,
                   layernorm_dst,
                   mean_d,
                   variance_d,
                   begin_norm_axis,
                   eps);
      }
    }
  } else {
    PADDLE_THROW(
        common::errors::Fatal("The TransLayerNormPluginDynamic TRT Plugin's "
                              "input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
