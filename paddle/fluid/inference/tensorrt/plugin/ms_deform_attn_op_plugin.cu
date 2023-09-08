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

#include "paddle/fluid/inference/tensorrt/plugin/ms_deform_attn_op_plugin.h"
#include <cub/cub.cuh>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

nvinfer1::DimsExprs MsDeformAttnPluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputDims,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
//     value_shape,
//     sampling_locations_shape,
//     attention_weights_shape,
//     spatial_shapes_shape,
//     level_start_index_shape) {
//   {{value_shape[0], sampling_locations_shape[1], value_shape[2] * value_shape[3]}};

  nvinfer1::DimsExprs output;
  output.nbDims = 3;
  output.d[0] = inputDims[0].d[0];
  output.d[1] = inputDims[1].d[1];
  output.d[2] = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                    *inputDims[0].d[2],
                                    *inputDims[0].d[3]);
  return output;
}

bool MsDeformAttnPluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos == 1) {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos == 2) {
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos == 3) {
      return (in.type == nvinfer1::DataType::kINT32) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos == 4) {
      return (in.type == nvinfer1::DataType::kINT32) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  return true;
}

nvinfer1::DataType MsDeformAttnPluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

int MsDeformAttnPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

int MsDeformAttnPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {

auto value = inputs[0];
auto sampling_locations = inputs[1];
auto attention_weights = inputs[2];
auto spatial_shapes = inputs[3];
auto level_start_index = inputs[4];

const int batch = input_desc[0].dims.d[0];
const int spatial_size = input_desc[0].dims.d[1];
const int num_heads = input_desc[0].dims.d[2];
const int channels = input_desc[0].dims.d[3];

const int num_levels = input_desc[3].dims.d[0];
const int num_query = input_desc[1].dims.d[1];
const int num_point = input_desc[1].dims.d[4];

const int im2col_step_ = std::min(batch, im2col_step_);

//   PD_CHECK(batch % im2col_step_ == 0, "batch(", batch,
//            ") must divide im2col_step(", im2col_step_, ")");

//   auto output = paddle::full({batch, num_query, num_heads * channels}, 0,
//                              value.type(), paddle::GPUPlace());

//   auto per_value_size = spatial_size * num_heads * channels;
//   auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
//   auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
//   auto per_output_size = num_query * num_heads * channels;

//   for (int n = 0; n < batch / im2col_step_; ++n) {
//     const int num_kernels = im2col_step_ * per_output_size;
//     const int num_actual_kernels = im2col_step_ * per_output_size;
//     const int num_threads = CUDA_NUM_THREADS;

//     ms_deformable_im2col_gpu_kernel<float>
//         <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
//            value.stream()>>>(
//             num_kernels,
//             value.data<float>() + n * im2col_step_ * per_value_size,
//             spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(),
//             sampling_locations.data<float>() +
//                 n * im2col_step_ * per_sample_loc_size,
//             attention_weights.data<float>() +
//                 n * im2col_step_ * per_attn_weight_size,
//             im2col_step_, spatial_size, num_heads, channels, num_levels,
//             num_query, num_point,
//             output.data<float>() + n * im2col_step_ * per_output_size);
//   }


      return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
