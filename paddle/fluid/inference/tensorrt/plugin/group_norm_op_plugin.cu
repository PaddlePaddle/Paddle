/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"
#include "paddle/phi/kernels/group_norm_kernel.h"

#include <cub/cub.cuh>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
using DataLayout = phi::DataLayout;

static inline int32_t divUp(int32_t m, int32_t n) { return (m + n - 1) / n; }

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sumSq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const &a,
                                         GroupSums const &b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

static int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <int tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(const GroupNormNHWCParams params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[tTHREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The offset.
    int64_t offset = static_cast<int64_t>(ni) * params.hwc +
                     static_cast<int64_t>(hwi) * params.c + ci;

    // Fetch two channels per thread.
    __half2 h2(0, 0);
    if (ci < params.c) {
      h2 = *reinterpret_cast<__half2 const *>(&params.srcX[offset]);
    }

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    // Update the sum.
    sum += f2.x + f2.y;
    // Update the sum of squares.
    sumSq += f2.x * f2.x + f2.y * f2.y;
  }

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = threadIdx.x * 2 / params.cPerGroup;
  int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced
  // stores later).
  // 2 channels per thread
  if (cj == params.cPerGroup - 2) {
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups) {
    return;
  }

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];

  // Store to global memory.
  atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
  atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNHWCSum(const GroupNormNHWCParams &params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params);
      break;
    case 480:
      groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params);
      break;
  }
}

template <int tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(const GroupNormNHWCParams params) {
  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < params.groups) {
    sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
  }

  // Load gamma/beta.
  float2 gammaF2, betaF2;
  if (ci < params.c) {
    gammaF2 = __half22float2(*reinterpret_cast<half2 const *>(
        reinterpret_cast<half const *>(params.gamma) + ci));
    betaF2 = __half22float2(*reinterpret_cast<half2 const *>(
        reinterpret_cast<half const *>(params.beta) + ci));
  }

  // Compute the mean.
  float mean = sum * params.invHWC;
  // Compute the variance.
  float var = sumSq * params.invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = rsqrtf(var + params.eps);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)ni * params.hwc + hwi * params.c + ci;

    // Fetch two channels per thread.
    __half2 h2(0, 0);
    if (ci < params.c) {
      h2 = *reinterpret_cast<__half2 const *>(&params.srcX[offset]);
    }

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Swish if needed.
    if (params.withSwish) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }

    // Store the scaled values.
    if (ci < params.c) {
      *reinterpret_cast<__half2 *>(&params.dst[offset]) = __float22half2_rn(f2);
    }
  }
}

void groupNormNHWCScale(const GroupNormNHWCParams &params,
                        cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params);
      break;
    case 480:
      groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params);
      break;
    default:
      PADDLE_THROW(
          platform::errors::Fatal("The function groupNormNHWCScale of "
                                  "GroupNorm TRT Plugin encounter error"));
  }
}

int GroupNormPlugin::initialize() TRT_NOEXCEPT {
  if (!with_fp16_) {
    // if use fp32
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_.data(),
               scale_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_.data(),
               bias_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    // if use fp16
    std::vector<half> scale_half(scale_.size());
    std::vector<half> bias_half(bias_.size());
    for (int i = 0; i < scale_.size(); ++i) {
      scale_half[i] = static_cast<half>(scale_[i]);
    }
    for (int i = 0; i < bias_.size(); ++i) {
      bias_half[i] = static_cast<half>(bias_[i]);
    }
    cudaMalloc(&scale_gpu_, sizeof(half) * scale_half.size());
    cudaMalloc(&bias_gpu_, sizeof(half) * bias_half.size());
    cudaMemcpy(scale_gpu_,
               scale_half.data(),
               scale_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_half.data(),
               bias_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  return 0;
}

bool GroupNormPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
    return ((type == nvinfer1::DataType::kHALF) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  } else {
    return ((type == nvinfer1::DataType::kFLOAT) &&
            (format == nvinfer1::PluginFormat::kLINEAR));
  }
}

nvinfer1::Dims GroupNormPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) TRT_NOEXCEPT {
  return inputDims[0];
}

int GroupNormPlugin::enqueue(int batch_size,
                             const void *const *inputs,
#if IS_TRT_VERSION_LT(8000)
                             void **outputs,
                             void *workspace,
#else
                             void *const *outputs,
                             void *workspace,
#endif
                             cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = this->getInputDims(0);
  int groups = groups_;
  float eps = eps_;
  std::vector<int> input_shape;
  input_shape.push_back(batch_size);
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }
  const auto input_ddim = phi::make_ddim(input_shape);

  int C = input_shape[1];

  PADDLE_ENFORCE_EQ(
      C,
      scale_.size(),
      platform::errors::InvalidArgument(
          "scale's size should be equal to the channel number in groupnorm,"
          "but got channel number:%d, scale's size:%d.",
          C,
          scale_.size()));
  PADDLE_ENFORCE_EQ(
      C,
      bias_.size(),
      platform::errors::InvalidArgument(
          "bias's size should be equal to the channel number in groupnorm,"
          "but got channel number:%d, bias's size:%d.",
          C,
          bias_.size()));
  float *mean_d = static_cast<float *>(workspace);
  float *variance_d = mean_d + input_shape[0] * groups_;
  float *temp_variance_d = variance_d + input_shape[0] * groups_;
  auto input_type = getDataType();
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp32";
    const float *input = static_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<float *>(bias_gpu_),
               reinterpret_cast<float *>(scale_gpu_),
               temp_variance_d,
               groups_,
               eps_,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp16";
    const half *input = static_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<half, float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<const half *>(bias_gpu_),
               reinterpret_cast<const half *>(scale_gpu_),
               temp_variance_d,
               groups_,
               eps_,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The GroupNorm TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}
nvinfer1::DimsExprs GroupNormPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputDims[0];
}

bool GroupNormPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of groupnorm plugin shoule not be nullptr."));
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
      return ((in.type == nvinfer1::DataType::kHALF) &&
              (in.format == nvinfer1::PluginFormat::kLINEAR ||
               in.format == nvinfer1::PluginFormat::kHWC8));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType GroupNormPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    platform::errors::InvalidArgument(
                        "The groupnorm Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT ||
                     input_types[0] == nvinfer1::DataType::kHALF),
                    true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));

  return input_types[0];
}
int GroupNormPluginDynamic::initialize() TRT_NOEXCEPT {
  if (with_fp16_ == false) {
    // if use fp32
    cudaMalloc(&scale_gpu_, sizeof(float) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(float) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_.data(),
               scale_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_.data(),
               bias_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  } else {
    // if use fp16
    std::vector<half> scale_half(scale_.size());
    std::vector<half> bias_half(bias_.size());
    for (int i = 0; i < scale_.size(); ++i) {
      scale_half[i] = static_cast<half>(scale_[i]);
    }
    for (int i = 0; i < bias_.size(); ++i) {
      bias_half[i] = static_cast<half>(bias_[i]);
    }
    cudaMalloc(&scale_gpu_, sizeof(half) * scale_.size());
    cudaMalloc(&bias_gpu_, sizeof(half) * bias_.size());
    cudaMemcpy(scale_gpu_,
               scale_half.data(),
               scale_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_,
               bias_half.data(),
               bias_half.size() * sizeof(half),
               cudaMemcpyHostToDevice);
  }
  return 0;
}

int GroupNormPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const auto &input_dims = input_desc[0].dims;
  int groups = groups_;
  float eps = eps_;

  std::vector<int> input_shape;
  for (int i = 0; i < input_dims.nbDims; i++) {
    input_shape.push_back(input_dims.d[i]);
  }

  const auto input_ddim = phi::make_ddim(input_shape);

  int C = input_shape[1];
  int image_size = input_shape[2] * input_shape[3];
  int batchSize = input_shape[0];

  PADDLE_ENFORCE_EQ(
      C,
      scale_.size(),
      platform::errors::InvalidArgument(
          "scale's size should be equal to the channel number in groupnorm,"
          "but got feature_size:%d, scale's size:%d.",
          C,
          scale_.size()));
  PADDLE_ENFORCE_EQ(
      C,
      bias_.size(),
      platform::errors::InvalidArgument(
          "bias's size should be equal to the channel number in groupnorm,"
          "but got feature_size:%d, bias's size:%d.",
          C,
          bias_.size()));

  float *mean_d = static_cast<float *>(workspace);
  float *variance_d = mean_d + input_shape[0] * groups_;
  float *temp_variance_d = variance_d + input_shape[0] * groups_;
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp32";
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    phi::GroupNormDirectCUDAFunctor<float, float> group_norm;
    group_norm(stream,
               input,
               input_shape,
               reinterpret_cast<float *>(bias_gpu_),
               reinterpret_cast<float *>(scale_gpu_),
               temp_variance_d,
               groups,
               eps,
               output,
               mean_d,
               variance_d,
               DataLayout::kNCHW);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. GroupNorm-->fp16";
    const half *input = reinterpret_cast<const half *>(inputs[0]);
    half *output = static_cast<half *>(outputs[0]);
    if (input_desc[0].format == nvinfer1::PluginFormat::kLINEAR) {
      phi::GroupNormDirectCUDAFunctor<half, float> group_norm;
      group_norm(stream,
                 input,
                 input_shape,
                 reinterpret_cast<half *>(bias_gpu_),
                 reinterpret_cast<half *>(scale_gpu_),
                 temp_variance_d,
                 groups,
                 eps,
                 output,
                 mean_d,
                 variance_d,
                 DataLayout::kNCHW);
    } else if (input_desc[0].format == nvinfer1::PluginFormat::kHWC8) {
      int32_t cPerBlock = 320;
      int32_t maxBlocksPerHW = 1024;
      switch (input_desc[0].dims.d[1]) {
        case 960:
        case 1920:
          cPerBlock = 480;
          break;
        case 512:
        case 256:
          cPerBlock = 256;
          break;
        case 128:
          cPerBlock = 128;
          break;
        default:
          cPerBlock = 320;
      }

      params_.withSwish = false;
      params_.dst = static_cast<half *>(outputs[0]);
      params_.srcX = static_cast<half const *>(inputs[0]);
      params_.gamma = scale_gpu_;
      params_.beta = bias_gpu_;
      params_.redBuffer = static_cast<float *>(workspace);
      params_.n = input_desc[0].dims.d[0];
      params_.h = input_desc[0].dims.d[2];
      params_.w = input_desc[0].dims.d[3];
      params_.c = input_desc[0].dims.d[1];
      params_.groups = groups_;
      params_.hw = params_.h * params_.w;
      const int32_t blocksPerHW = findMaxDivisor(params_.hw, maxBlocksPerHW);
      params_.hwPerBlock = divUp(params_.hw, blocksPerHW);
      params_.cPerBlock = cPerBlock;
      params_.cPerGroup = params_.c / params_.groups;
      params_.hwc = params_.hw * params_.c;
      params_.invHWC = 1.F / static_cast<float>(params_.hw * params_.cPerGroup);
      params_.groupsPerBlock = cPerBlock / params_.cPerGroup;
      params_.eps = eps_;

      cudaMemsetAsync(params_.redBuffer,
                      0,
                      2 * sizeof(float) * params_.n * groups_,
                      stream);
      groupNormNHWCSum(params_, stream);
      groupNormNHWCScale(params_, stream);
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "The Groupnorm TRT Plugin's only support nchw or nhwc8 input"));
    }
  } else {
    // input not float
    PADDLE_THROW(platform::errors::Fatal(
        "The Groupnorm TRT Plugin's only support fp32 or fp16 input"));
  }

  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
