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

#include <stdio.h>
#include <cassert>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/conv2d_transpose_plugin.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedFilterDescriptor = platform::ScopedFilterDescriptor;
using ScopedConvolutionDescriptor = platform::ScopedConvolutionDescriptor;
using DataLayout = platform::DataLayout;

nvinfer1::Dims Conv2dTransposePlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *inputDims, int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims const &input_dims = inputDims[0];
  assert(input_dims.nbDims == 3);
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[0] = output_shape_[0];
  output_dims.d[1] = output_shape_[1];
  output_dims.d[2] = output_shape_[2];
  return output_dims;
}

int Conv2dTransposePlugin::enqueue(int batchSize, const void *const *inputs,
                                   void **outputs, void *workspace,
                                   cudaStream_t stream) {
  const float *input_data = reinterpret_cast<const float *>(inputs[0]);
  float *output_data = reinterpret_cast<float **>(outputs)[0];

  ScopedTensorDescriptor input_desc;
  ScopedTensorDescriptor output_desc;
  ScopedFilterDescriptor filter_desc;
  ScopedConvolutionDescriptor conv_desc;
  DataLayout layout = DataLayout::kNCHW;

  std::vector<int> input_shape = input_shape_;
  std::vector<int> output_shape = output_shape_;
  std::vector<int> filter_shape = filter_shape_;

  input_shape.insert(input_shape.begin(), batchSize);
  output_shape.insert(output_shape.begin(), batchSize);

  int input_num = accumulate(input_shape.begin(), input_shape.end(), 0);
  int output_num = accumulate(output_shape.begin(), output_shape.end(), 0);
  int filter_num = accumulate(filter_shape.begin(), filter_shape.end(), 0);

  // Input: (N, M, H, W)
  cudnnTensorDescriptor_t cudnn_input_desc =
      input_desc.descriptor<float>(layout, input_shape, groups_);
  // Output: (N, C, O_h, O_w)
  cudnnTensorDescriptor_t cudnn_output_desc =
      output_desc.descriptor<float>(layout, output_shape, groups_);
  // Filter (M, C, K_h, K_w)
  cudnnFilterDescriptor_t cudnn_filter_desc =
      filter_desc.descriptor<float>(layout, filter_shape, groups_);

  cudnnConvolutionDescriptor_t cudnn_conv_desc =
      conv_desc.descriptor<float>(paddings_, strides_, dilations_);

  size_t workspace_size_in_bytes;
  size_t workspace_size_limit = 4096 * 1024 * 1024;

  platform::CUDAPlace place(gpu_id_);

  platform::CudnnHolder cudnn_holder(&stream, place);
  platform::CudnnWorkspaceHandle workspace_handle(&cudnn_holder);
  cudnnHandle_t cudnn_handle = cudnn_holder.cudnn_handle();

  cudnnConvolutionBwdDataAlgo_t algo;
  // Get the algorithm
  CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
      // dxDesc: Handle to the previously initialized output tensor
      // descriptor.
      cudnn_output_desc, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
      workspace_size_limit, &algo));

  // get workspace size able to allocate
  CUDNN_ENFORCE(platform::dynload::cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_handle, cudnn_filter_desc, cudnn_input_desc, cudnn_conv_desc,
      cudnn_output_desc, algo, &workspace_size_in_bytes));

  int input_offset = input_num / input_shape_[0] / groups_;
  int output_offset = output_num / output_shape_[0] / groups_;
  int filter_offset = filter_num / groups_;
  float alpha = 1.0f, beta = 0.0f;
  for (int g = 0; g < groups_; g++) {
    auto cudnn_func = [&](void *cudnn_workspace) {
      CUDNN_ENFORCE(platform::dynload::cudnnConvolutionBackwardData(
          cudnn_handle, &alpha, cudnn_filter_desc,
          weight_data_ + filter_offset * g, cudnn_input_desc,
          input_data + input_offset * g, cudnn_conv_desc, algo, cudnn_workspace,
          workspace_size_in_bytes, &beta, cudnn_output_desc,
          output_data + output_offset * g));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size_in_bytes);
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
