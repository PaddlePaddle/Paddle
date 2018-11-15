/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include "paddle/fluid/inference/tensorrt/plugin/elementwise_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

nvinfer1::Dims ElementWisePlugin::getOutputDimensions(
    int index, const nvinfer1::Dims *input_dims, int num_inputs) {
  PADDLE_ENFORCE_EQ(index, 0);
  PADDLE_ENFORCE_EQ(num_inputs, 2);
  PADDLE_ENFORCE_NOT_NULL(input_dims);
  return input_dims[0];
}

int ElementWisePlugin::initialize() {
  PADDLE_ENFORCE_GT(dims_y_.nbDims, 0);

  axis_ = (axis_ == -1) ? dims_x_.nbDims - dims_y_.nbDims : axis_;
  int trimed_nb_dims = dims_y_.nbDims;
  for (; trimed_nb_dims > 0; --trimed_nb_dims) {
    if (dims_y_.d[trimed_nb_dims - 1] != 1) {
      break;
    }
  }
  dims_y_.nbDims = trimed_nb_dims;

  PADDLE_ENFORCE_GE(dims_x_.nbDims, dims_y_.nbDims + axis_);
  PADDLE_ENFORCE_LT(axis_, dims_x_.nbDims);

  prev_size_ = 1;
  midd_size_ = 1;
  post_size_ = 1;
  for (int i = 0; i < axis_; ++i) {
    prev_size_ *= dims_x_.d[i];
  }

  for (int i = 0; i < dims_y_.nbDims; ++i) {
    PADDLE_ENFORCE_EQ(dims_x_.d[i + axis_], dims_y_.d[i],
                      "Broadcast dimension mismatch.");
    midd_size_ *= dims_y_.d[i];
  }

  for (int i = axis_ + dims_y_.nbDims; i < dims_x_.nbDims; ++i) {
    post_size_ *= dims_x_.d[i];
  }
  LOG(INFO) << "prev_size_: " << prev_size_ << ", midd_size_: " << midd_size_
            << ", post_size_: " << post_size_;
  return 0;
}

int ElementWisePlugin::enqueue(int batch_size, const void *const *inputs,
                               void **outputs, void *workspace,
                               cudaStream_t stream) {
  return 0;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
