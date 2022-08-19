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

#include <cassert>
#include <cstring>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/cast_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename Tin, typename Tout>
__global__ void cast_kernel(const Tin* input, Tout* output, int num) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    output[idx] = static_cast<Tout>(input[idx]);
  }
}

nvinfer1::Dims CastPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims* input_dims,
                                               int num_inputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(num_inputs,
                    1,
                    platform::errors::InvalidArgument(
                        "Invalid number of inputs of split TRT plugin. "
                        "Expected 1, received %d.",
                        num_inputs));
  PADDLE_ENFORCE_LT(
      index,
      this->getNbOutputs(),
      platform::errors::InvalidArgument(
          "Index of output should be less than the total number of outputs in "
          "split TensorRT plugin. Received index = %d >= total outputs = %d",
          index,
          this->getNbOutputs()));

  nvinfer1::Dims output_dims = input_dims[0];
  return output_dims;
}

int CastPlugin::initialize() TRT_NOEXCEPT { return 0; }

// nothing to release according to initialize
void CastPlugin::terminate() TRT_NOEXCEPT {}

int CastPlugin::enqueue(int batch_size,
                        const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                        void** outputs,
                        void* workspace,
                        cudaStream_t stream) {
#else
                        void* const* outputs,
                        void* workspace,
                        cudaStream_t stream) TRT_NOEXCEPT {
#endif
  const auto& input_dims = this->getInputDims(0);
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }

  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  // 0 : bool
  // 2 : int
  // 4 : half
  // 5 : float

  if (intype_ == 2 && outtype_ == 5) {  // int -> float
    const int* input = static_cast<const int*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<int, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 2) {  // float -> int
    const float* input = static_cast<const float*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<float, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 2) {  // bool -> int
    const bool* input = static_cast<const bool*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<bool, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 2 && outtype_ == 0) {  // int -> bool
    const int* input = static_cast<const int*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<int, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 5) {  // bool -> float
    const bool* input = static_cast<const bool*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<bool, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 0) {  // float -> bool
    const float* input = static_cast<const float*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<float, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 5) {  // float -> float
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<float, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 2 && outtype_ == 2) {  // int -> int
    const int* input = static_cast<const int*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<int, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 0) {  // bool -> bool
    const bool* input = static_cast<const bool*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<bool, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 0) {  // half -> bool
    const half* input = static_cast<const half*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<half, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 2) {  // half -> int
    const half* input = static_cast<const half*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<half, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 5) {  // half -> float
    const half* input = static_cast<const half*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<half, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 4) {  // half -> half
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    cast_kernel<half, half>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  }
  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

nvinfer1::DimsExprs CastPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  return inputs[0];
}

bool CastPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
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

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (fp16_) {
    return in.type == nvinfer1::DataType::kFLOAT ||
           in.type == nvinfer1::DataType::kINT32 ||
           in.type == nvinfer1::DataType::kBOOL;

  } else {
    return in.type == nvinfer1::DataType::kFLOAT ||
           in.type == nvinfer1::DataType::kINT32 ||
           in.type == nvinfer1::DataType::kBOOL;
  }
}

nvinfer1::DataType CastPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    platform::errors::InvalidArgument(
                        "The Gelu Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));

  if (outtype_ == 0) {
    return nvinfer1::DataType::kBOOL;
  } else if (outtype_ == 2) {
    return nvinfer1::DataType::kINT32;
  } else {
    return nvinfer1::DataType::kFLOAT;
  }
}

int CastPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                               const nvinfer1::PluginTensorDesc* output_desc,
                               const void* const* inputs,
                               void* const* outputs,
                               void* workspace,
                               cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  size_t num = ProductDim(input_dims);
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;
  // 0 : bool
  // 2 : int
  // 4 : half
  // 5 : float

  if (intype_ == 2 && outtype_ == 5) {  // int -> float
    const int* input = static_cast<const int*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<int, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 2) {  // float -> int
    const float* input = static_cast<const float*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<float, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 2) {  // bool -> int
    const bool* input = static_cast<const bool*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<bool, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 2 && outtype_ == 0) {  // int -> bool
    const int* input = static_cast<const int*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<int, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 5) {  // bool -> float
    const bool* input = static_cast<const bool*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<bool, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 0) {  // float -> bool
    const float* input = static_cast<const float*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<float, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 5 && outtype_ == 5) {  // float -> float
    const float* input = static_cast<const float*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<float, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 2 && outtype_ == 2) {  // int -> int
    const int* input = static_cast<const int*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<int, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 0 && outtype_ == 0) {  // bool -> bool
    const bool* input = static_cast<const bool*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<bool, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 0) {  // half -> bool
    const half* input = static_cast<const half*>(inputs[0]);
    bool* output = static_cast<bool*>(outputs[0]);
    cast_kernel<half, bool>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 2) {  // half -> int
    const half* input = static_cast<const half*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    cast_kernel<half, int>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 5) {  // half -> float
    const half* input = static_cast<const half*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<half, float>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  } else if (intype_ == 4 && outtype_ == 4) {  // half -> half
    const half* input = static_cast<const half*>(inputs[0]);
    half* output = static_cast<half*>(outputs[0]);
    cast_kernel<half, half>
        <<<grid_size, block_size, 0, stream>>>(input, output, num);
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
