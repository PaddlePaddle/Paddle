/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
using DataLayout = framework::DataLayout;

int GroupNormPlugin::initialize() TRT_NOEXCEPT { return 0; }

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

  int device_id;
  cudaGetDevice(&device_id);
  const float *input = reinterpret_cast<const float *>(inputs[0]);
  float *output = static_cast<float *>(outputs[0]);

  scale_t.Resize(phi::make_ddim({C}));
  bias_t.Resize(phi::make_ddim({C}));

  mean_t.Resize(phi::make_ddim(mean_shape_));
  variance_t.Resize(phi::make_ddim(variance_shape_));
  float *scale_d = scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
  float *variance_d =
      variance_t.mutable_data<float>(platform::CUDAPlace(device_id));

  framework::Tensor temp_variance_t;
  temp_variance_t.Resize(phi::make_ddim(variance_shape_));
  float *temp_variance_d =
      temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
  cudaMemcpyAsync(scale_d,
                  scale_.data(),
                  sizeof(float) * C,
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(
      bias_d, bias_.data(), sizeof(float) * C, cudaMemcpyHostToDevice, stream);
  const int group_size = C / groups_;
  // printf("static group_size: %d\r\n", group_size);
  const int W = input_ddim[input_ddim.size() - 1];
  int image_size = 1;
  for (int i = 2; i < input_ddim.size(); ++i) {
    image_size *= input_ddim[i];
  }
  int block_size = std::min(1024, image_size);
  dim3 grid(group_size, groups_, input_ddim[0]);
  dim3 threads(block_size, 1, 1);

  using AccT = typename phi::kps::details::MPTypeTrait<float>::Type;
  constexpr int vec_size = sizeof(float4) / sizeof(float);
  int size = group_size * image_size;  // group element size
  const int max_num_threads = 1024;
  int max_block_size = std::min(size / vec_size, max_num_threads);
  int block_size_nchw = 1;
  while (block_size_nchw < max_block_size) {
    block_size_nchw *= 2;
  }

  block_size_nchw = std::max(block_size_nchw, phi::kps::details::kWarpSize);
  dim3 grids(input_ddim[0] * groups_);
  dim3 blocks(block_size_nchw);

  if (size < vec_size * block_size_nchw) {
    phi::ScalarGetMeanAndVarNCHW<float>
        <<<grids, blocks, 0, stream>>>(input, mean_d, temp_variance_d, size);
  } else {
    phi::VectorizedGetMeanAndVarNCHW<float, AccT, vec_size>
        <<<grids, blocks, 0, stream>>>(input, mean_d, temp_variance_d, size);
  }
  phi::GroupNormForward<float, 3><<<grid, threads, 0, stream>>>(
      input,
      mean_d,
      temp_variance_d,
      scale_d,
      bias_d,
      input_ddim[0],
      C,
      W,
      image_size,
      groups_,
      group_size,
      eps_,
      output,
      variance_d,
      DataLayout::kNCHW  // for now, we only support nchw for group norm
  );
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
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
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
  return input_types[0];
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
  int batchSize = input_shape[0];
  std::vector<int64_t> batched_mean_shape = {batchSize};
  batched_mean_shape.insert(
      batched_mean_shape.end(), mean_shape_.begin(), mean_shape_.end());
  std::vector<int64_t> batched_variance_shape = {batchSize};
  batched_variance_shape.insert(batched_variance_shape.end(),
                                variance_shape_.begin(),
                                variance_shape_.end());
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

  int device_id;
  cudaGetDevice(&device_id);
  auto input_type = input_desc[0].type;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    const float *input = reinterpret_cast<const float *>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    scale_t.Resize(phi::make_ddim({C}));
    bias_t.Resize(phi::make_ddim({C}));

    mean_t.Resize(phi::make_ddim(batched_mean_shape));
    variance_t.Resize(phi::make_ddim(batched_variance_shape));
    float *scale_d =
        scale_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *bias_d = bias_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *mean_d = mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *variance_d =
        variance_t.mutable_data<float>(platform::CUDAPlace(device_id));

    framework::Tensor temp_mean_t;
    framework::Tensor temp_variance_t;
    temp_mean_t.Resize(phi::make_ddim(batched_mean_shape));
    temp_variance_t.Resize(phi::make_ddim(batched_variance_shape));
    float *temp_mean_d =
        temp_mean_t.mutable_data<float>(platform::CUDAPlace(device_id));
    float *temp_variance_d =
        temp_variance_t.mutable_data<float>(platform::CUDAPlace(device_id));
    cudaMemcpyAsync(scale_d,
                    scale_.data(),
                    sizeof(float) * C,
                    cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(bias_d,
                    bias_.data(),
                    sizeof(float) * C,
                    cudaMemcpyHostToDevice,
                    stream);
    const int group_size = C / groups_;
    const int W = input_ddim[input_ddim.size() - 1];
    int image_size = 1;
    for (int i = 2; i < input_ddim.size(); ++i) {
      image_size *= input_ddim[i];
    }
    int block_size = std::min(1024, image_size);
    dim3 grid(group_size, groups_, input_ddim[0]);
    dim3 threads(block_size, 1, 1);

    using AccT = typename phi::kps::details::MPTypeTrait<float>::Type;
    constexpr int vec_size = sizeof(float4) / sizeof(float);
    int size = group_size * image_size;  // group element size
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }

    block_size_nchw = std::max(block_size_nchw, phi::kps::details::kWarpSize);
    dim3 grids(input_ddim[0] * groups_);
    dim3 blocks(block_size_nchw);

    if (size < vec_size * block_size_nchw) {
      phi::ScalarGetMeanAndVarNCHW<float><<<grids, blocks, 0, stream>>>(
          input, temp_mean_d, temp_variance_d, size);
    } else {
      phi::VectorizedGetMeanAndVarNCHW<float, AccT, vec_size>
          <<<grids, blocks, 0, stream>>>(
              input, temp_mean_d, temp_variance_d, size);
    }
    phi::GroupNormForward<float, 3><<<grid, threads, 0, stream>>>(
        input,
        temp_mean_d,
        temp_variance_d,
        scale_d,
        bias_d,
        input_ddim[0],
        C,
        W,
        image_size,
        groups_,
        group_size,
        eps_,
        output,
        variance_d,
        DataLayout::kNCHW  // for now, we only support nchw for group norm
    );

  } else {
    // input not float
    PADDLE_THROW(platform::errors::Fatal(
        "The Groupnorm TRT Plugin's only support fp32 input"));
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
