/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/preln_groupnorm_act_op_plugin.h"
#include <cub/cub.cuh>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
nvinfer1::DimsExprs PrelnGroupnormActPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  return inputDims[0];
}

bool PrelnGroupnormActPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of prelnGroupnormAct plugin shoule not be nullptr."));
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
              (in.format == nvinfer1::PluginFormat::kHWC8));
    } else {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
    }
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType PrelnGroupnormActPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

int PrelnGroupnormActPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

static inline int32_t divUp(int32_t m, int32_t n) { return (m + n - 1) / n; }

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

template <int32_t tTHREADS_PER_BLOCK>
__global__ void prelnGroupNormNHWCSumKernel(GroupNormNHWCParams params) {
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
      // int64_t offsetY = static_cast<int64_t>(ni) * params.c + ci;
      __half2 y = *reinterpret_cast<__half2 const *>(&params.srcY[offset]);
      h2 = *reinterpret_cast<__half2 const *>(&params.srcX[offset]);
      h2 = __hadd2(h2, y);
      // elementwise_add
      *reinterpret_cast<__half2 *>(&params.eleOut[offset]) = h2;
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
  if (cj == params.cPerGroup - 2 /* 2 channels per thread */) {
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

void prelnGroupNormNHWCSum(GroupNormNHWCParams const &params,
                           cudaStream_t stream) {
  // Make sure the values are as we expect.
  PADDLE_ENFORCE_EQ(params.c % params.cPerBlock,
                    0,
                    platform::errors::InvalidArgument(
                        "The groupNormNHWCSum of prelnGroupnormAct Plugin got "
                        "wrong parameters"
                        "params.c %% params.cPerBlock should be 0, but get %d.",
                        params.c % params.cPerBlock));
  PADDLE_ENFORCE_EQ(
      params.hw % params.hwPerBlock,
      0,
      platform::errors::InvalidArgument(
          "The groupNormNHWCSum of prelnGroupnormAct Plugin got wrong "
          "parameters"
          "params.hw  %% params.hwPerBlock should be 0, but get %d.",
          params.hw % params.hwPerBlock));
  // Make sure a group does not span multiple blocks.
  PADDLE_ENFORCE_EQ(
      params.cPerBlock % params.cPerGroup,
      0,
      platform::errors::InvalidArgument(
          "The groupNormNHWCSum of prelnGroupnormAct Plugin got wrong "
          "parameters"
          "params.cPerBlock %% params.cPerGroup should be 0, but get %d.",
          params.cPerBlock % params.cPerGroup));
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      prelnGroupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params);
      break;
    case 480:
      prelnGroupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      prelnGroupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      prelnGroupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params);
      break;
    default:
      PADDLE_THROW(platform::errors::Fatal(
          "The function groupNormNHWCSum of prelnGroupnormAct TRT Plugin "
          "encounter error"));
  }
}

template <int32_t tTHREADS_PER_BLOCK>
__global__ void prelnGroupNormNHWCScaleKernel(GroupNormNHWCParams params) {
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
    gammaF2 = *reinterpret_cast<float2 const *>(
        reinterpret_cast<float const *>(params.gamma) + ci);
    betaF2 = *reinterpret_cast<float2 const *>(
        reinterpret_cast<float const *>(params.beta) + ci);
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
      h2 = *reinterpret_cast<__half2 const *>(&params.eleOut[offset]);
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

void prelnGroupNormNHWCScale(GroupNormNHWCParams const &params,
                             cudaStream_t stream) {
  // Make sure the dimensions are aligned with what we expect.
  PADDLE_ENFORCE_EQ(
      params.c % params.cPerBlock,
      0,
      platform::errors::InvalidArgument(
          "The groupNormNHWCScale of prelnGroupnormAct Plugin got "
          "wrong parameters"
          "params.c %% params.cPerBlock should be 0, but get %d.",
          params.c % params.cPerBlock));
  // Make sure a group does not span multiple blocks.
  PADDLE_ENFORCE_EQ(
      params.cPerBlock % params.cPerGroup,
      0,
      platform::errors::InvalidArgument(
          "The groupNormNHWCScale of prelnGroupnormAct Plugin got wrong "
          "parameters"
          "params.cPerBlock %% params.cPerGroup should be 0, but get %d.",
          params.cPerBlock % params.cPerGroup));
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      prelnGroupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params);
      break;
    case 480:
      prelnGroupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params);
      break;
    case 256:
      prelnGroupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params);
      break;
    case 128:
      prelnGroupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params);
      break;
    default:
      PADDLE_THROW(platform::errors::Fatal(
          "The function groupNormNHWCSum of prelnGroupnormAct TRT Plugin "
          "encounter error"));
  }
}

int PrelnGroupnormActPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. prelnGroupnormAct-->fp32";
    PADDLE_THROW(platform::errors::Fatal(
        "The prelnGroupnormAct TRT Plugin's only support fp16 input"));
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. prelnGroupnormAct-->fp16";

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
    params_.withSwish = true;
    params_.dst = static_cast<half *>(outputs[1]);
    params_.eleOut = static_cast<half *>(outputs[0]);
    params_.srcX = static_cast<half const *>(inputs[0]);
    params_.srcY = static_cast<half const *>(inputs[1]);
    params_.gamma = scale_gpu_.get();
    params_.beta = bias_gpu_.get();
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

    cudaMemsetAsync(params_.redBuffer, 0, ws_, stream);
    prelnGroupNormNHWCSum(params_, stream);
    prelnGroupNormNHWCScale(params_, stream);

  } else {
    // input not fp16
    PADDLE_THROW(platform::errors::Fatal(
        "The PrelnGroupnormAct TRT Plugin's only support fp16 input"));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
