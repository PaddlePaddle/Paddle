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
#include "paddle/fluid/inference/tensorrt/plugin/stack_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)
size_t StackPluginDynamic::getSerializationSize() const { return 0; }

void StackPluginDynamic::serialize(void* buffer) const {}

nvinfer1::DimsExprs StackPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  nvinfer1::DimsExprs output(inputs[0]);
  output.nbDims = inputs[0].nbDims + 1;

  for (int i = inputs[0].nbDims; i > axis_; --i) {
    output.d[i] = inputs[0].d[i - 1];
  }
  output.d[axis_] = expr_builder.constant(nb_inputs);
  return output;
}

bool StackPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of stack plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (pos == 0) {
#ifdef SUPPORTS_CUDA_FP16
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
  }
  const nvinfer1::PluginTensorDesc& prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType StackPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The index should be equal to 0"));
  return input_types[0];
}

template <typename T>
__global__ void StackKernel(const T* const* input, T* output, int num_stack,
                            int base_unit) {
  int stack_id = blockIdx.x;
  int lead_id = blockIdx.y;

  for (int i = threadIdx.x; i < base_unit; i += blockDim.x) {
    output[lead_id * num_stack * base_unit + stack_id * base_unit + i] =
        input[stack_id][lead_id * base_unit + i];
  }
}

int StackPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                const nvinfer1::PluginTensorDesc* output_desc,
                                const void* const* inputs, void* const* outputs,
                                void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;  // (batch, seq, seq)
  auto out_dims = output_desc[0].dims;   // (batch, num_head, seq, seq)
  auto out_num_dims = out_dims.nbDims;

  int base_unit = 1;
  for (int i = axis_ + 1; i < out_num_dims; ++i) {
    PADDLE_ENFORCE_GT(out_dims.d[i], 0,
                      platform::errors::InvalidArgument(
                          "Input dimensions should be greater than 0"));
    base_unit *= out_dims.d[i];
  }

  int lead_unit = 1;
  for (int i = 0; i < axis_; ++i) {
    PADDLE_ENFORCE_GT(out_dims.d[i], 0,
                      platform::errors::InvalidArgument(
                          "Input dimensions should be greater than 0"));
    lead_unit *= out_dims.d[i];
  }

  cudaMemcpyAsync(reinterpret_cast<void*>(in_ptr_gpu_),
                  reinterpret_cast<const void* const>(inputs),
                  sizeof(void*) * out_dims.d[axis_], cudaMemcpyHostToDevice,
                  stream);

  int num_stacks = out_dims.d[axis_];
  dim3 num_blocks(num_stacks, lead_unit);
  int num_threads = 256;
  auto infer_type = input_desc[0].type;

  if (infer_type == nvinfer1::DataType::kFLOAT) {
    float* output = static_cast<float*>(outputs[0]);
    StackKernel<float><<<num_blocks, num_threads, 0, stream>>>(
        reinterpret_cast<const float* const*>(in_ptr_gpu_), output, num_stacks,
        base_unit);
  } else if (infer_type == nvinfer1::DataType::kHALF) {
#ifdef SUPPORTS_CUDA_FP16
    __half* output = static_cast<__half*>(outputs[0]);
    StackKernel<__half><<<num_blocks, num_threads, 0, stream>>>(
        reinterpret_cast<const __half* const*>(in_ptr_gpu_), output, num_stacks,
        base_unit);
#else
    PADDLE_THROW(platform::errors::Fatal(
        "The cuda archs you specific should greater than 600."));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("The Stack TRT Plugin's input type only "
                                "support float or half currently."));
  }
  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
