// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstring>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/convert_mask_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

/* This plugin currently converts the matmul output [B, S, S]
to the mask with the bertQKV fused_multihead_attention format */

constexpr size_t threadsPerCta128 = 2 * 2 * 32;

constexpr size_t xmmasM128 = 4;

constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;

nvinfer1::DimsExprs ConvertMaskPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  auto cms128 = expr_builder.constant(packedMaskSize128);
  auto fp16maskSize = expr_builder.operation(
      nvinfer1::DimensionOperation::kPROD, *cms128, *expr_builder.constant(2));

  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = fp16maskSize;

  return ret;
}

bool ConvertMaskPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) {
  const nvinfer1::PluginTensorDesc& desc = in_out[pos];
  /*  input: [B, S, S] */
  /* output: [B, 2*maskSize] */
  assert(nb_inputs == 1);
  assert(nb_outputs == 1);

  if (pos == 0) {
    std::cerr << "desc.type: " << static_cast<int>(desc.type) << " "
              << desc.dims.nbDims << std::endl;
    return ((desc.type == nvinfer1::DataType::kFLOAT ||
             desc.type == nvinfer1::DataType::kHALF) &&
            desc.dims.nbDims == 3);
  }
  std::cerr << "output.type: " << static_cast<int>(desc.type) << " "
            << desc.dims.nbDims << std::endl;
  // return desc.type == nvinfer1::DataType::kHALF;
  return true;
}

nvinfer1::DataType ConvertMaskPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The convert mask plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return nvinfer1::DataType::kHALF;
}

template <typename T>
__global__ void CastToIntAndReduce(const T* input, int* output, int seq_len,
                                   int batch) {
  int bid = blockIdx.x;
  int sid = threadIdx.x;
  output[sid * batch + bid] =
      static_cast<int>(input[bid * seq_len * seq_len + sid]);
}

__global__ void fillSBSMaskKernel(const uint32_t warps_m,
                                  const uint32_t warps_n, const uint32_t S,
                                  const int* inputMaskSB,
                                  uint32_t* inputMaskX) {
  extern __shared__ int shm_mask[];  // S mask elements of this batch

  const size_t xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);
  const uint32_t threads_per_cta = blockDim.x;
  const uint32_t xmmas_m = gridDim.x;
  const uint32_t B = gridDim.y;

  const uint32_t mi = blockIdx.x;
  const uint32_t bi = blockIdx.y;
  const uint32_t tidx = threadIdx.x;

  const size_t warp = tidx / 32;
  const size_t warp_m = warp % warps_m;
  const size_t warp_n = warp / warps_m;
  const size_t lane = tidx % 32;
  const size_t col = warp_n * 16 + lane % 4 * 2;

  // load the mask corresponding to one batch
  for (uint32_t si = tidx; si < S; si += threads_per_cta) {
    // not coalesced to conform to current input format: SxB
    shm_mask[si] = inputMaskSB[si * B + bi];
  }
  __syncthreads();

  uint32_t mask = 0u;

  for (size_t ni = 0; ni < xmmas_n; ++ni) {
    const int offset = ni * 16 * warps_n + col;
    mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 0);
    mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 1);
    mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 2);
    mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 3);
    mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 4);
    mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 5);
    mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 6);
    mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 7);
  }

  inputMaskX[(bi * xmmas_m + mi) * threads_per_cta + tidx] = mask;
}

void convertMask(const uint32_t S, const uint32_t B, const uint32_t warps_m,
                 const uint32_t warps_n, const uint32_t warps_k,
                 const int* inputMaskSB, uint32_t* inputMaskX,
                 cudaStream_t stream) {
  const size_t xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);

  const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
  dim3 grid(xmmas_m, B);
  fillSBSMaskKernel<<<grid, threads_per_cta, S * sizeof(int), stream>>>(
      warps_m, warps_n, S, inputMaskSB, inputMaskX);
}

int ConvertMaskPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  auto output_dims = output_desc[0].dims;
  size_t num_elements = ProductDim(input_dims);
  size_t out_num_elements = ProductDim(output_dims);
  int batch = input_dims.d[0];
  int seq_len = input_dims.d[1];

  assert(num_elements == out_num_elements * seq_len);
  assert(seq_len <= 1024);
  assert(output_desc.type == nvinfer1::DataType::kHALF);

  // temp use, should remove
  int* inputMaskSB;
  cudaMalloc(&inputMaskSB, batch * seq_len * sizeof(int));

  if (input_desc[0].type == nvinfer1::DataType::kFLOAT) {
    CastToIntAndReduce<float><<<batch, seq_len, 0, stream>>>(
        static_cast<const float*>(inputs[0]), inputMaskSB, seq_len, batch);
  } else {
    CastToIntAndReduce<half><<<batch, seq_len, 0, stream>>>(
        static_cast<const half*>(inputs[0]), inputMaskSB, seq_len, batch);
  }

  assert(seq_len == 128);
  size_t warps_m = 0, warps_n = 0, warps_k = 1;
  if (seq_len == 128) {
    warps_m = 2;
    warps_n = 2;
  }

  convertMask(seq_len, batch, warps_m, warps_n, warps_k, inputMaskSB,
              static_cast<uint32_t*>(outputs[0]), stream);

  cudaFree(inputMaskSB);
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
