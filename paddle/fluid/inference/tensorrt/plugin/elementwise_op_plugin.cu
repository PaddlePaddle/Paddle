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
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

ElementWisePlugin *CreateElementWisePluginDeserialize(const void *buffer,
                                                      size_t length) {
  return new ElementWisePlugin(buffer, length);
}
REGISTER_TRT_PLUGIN("elementwise_plugin", CreateElementWisePluginDeserialize);

namespace details {
template <typename T>
struct Add {
  __device__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct Mul {
  __device__ T operator()(const T &a, const T &b) const { return a * b; }
};
}  // namespace details

template <typename T, typename Operator>
__global__ void elementwise_kernel(const size_t total, const T *x_data,
                                   const T *y_data, T *out_data, int pre, int n,
                                   int post, Operator op) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total) {
    int idx = tid / post % n;
#if __CUDA_ARCH__ >= 350
    out_data[tid] = op(__ldg(x_data + tid), __ldg(y_data + idx));
#else
    out_data[tid] = op(x_data[tid], y_data[idx]);
#endif
  }
}

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
  return 0;
}

int ElementWisePlugin::enqueue(int batch_size, const void *const *inputs,
                               void **outputs, void *workspace,
                               cudaStream_t stream) {
  const float *x = reinterpret_cast<const float *>(inputs[0]);
  const float *y = reinterpret_cast<const float *>(inputs[1]);
  float *out = reinterpret_cast<float *>(outputs[0]);

  int num = batch_size * prev_size_ * midd_size_ * post_size_;
  int thread = 256;
  int block = (num + thread - 1) / thread;
  if (type_ == "add") {
    elementwise_kernel<<<block, thread, 0, stream>>>(
        num, x, y, out, prev_size_, batch_size * midd_size_, post_size_,
        details::Add<float>());
  } else if (type_ == "mul") {
    elementwise_kernel<<<block, thread, 0, stream>>>(
        num, x, y, out, prev_size_, batch_size * midd_size_, post_size_,
        details::Mul<float>());
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "The %s type elementwise is not implemented in trt plugin.", type_));
  }

  return cudaGetLastError() != cudaSuccess;
}

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

int ElementwisePluginDynamic::initialize() { return 0; }

size_t ElementwisePluginDynamic::getSerializationSize() const { return 0; }

void ElementwisePluginDynamic::serialize(void *buffer) const {}

nvinfer1::DimsExprs ElementwisePluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) {
  return inputs[0];
}

bool ElementwisePluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
  // output
  return in.type == prev.type && in.format == prev.format;
}

nvinfer1::DataType ElementwisePluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The Elementwise Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int ElementwisePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) {
  auto x_dims = input_desc[0].dims;
  auto y_dims = input_desc[1].dims;
  int axis = (axis_ == -1) ? x_dims.nbDims - y_dims.nbDims : axis_;
  int batch_size = x_dims.d[0];

  int prev_size = 1;
  int midd_size = 1;
  int post_size = 1;
  for (int i = 0; i < axis; ++i) {
    prev_size *= x_dims.d[i];
  }

  int trimed_nb_dims = y_dims.nbDims;
  for (; trimed_nb_dims > 0; --trimed_nb_dims) {
    if (y_dims.d[trimed_nb_dims - 1] != 1) {
      break;
    }
  }

  for (int i = 0; i < trimed_nb_dims; ++i) {
    PADDLE_ENFORCE_EQ(x_dims.d[i + axis], y_dims.d[i],
                      platform::errors::InvalidArgument(
                          "Broadcast dimension mismatch found in trt "
                          "elementwise plugin's x and y input."));
    midd_size *= y_dims.d[i];
  }

  for (int i = axis + trimed_nb_dims; i < x_dims.nbDims; ++i) {
    post_size *= x_dims.d[i];
  }

  const float *x = static_cast<const float *>(inputs[0]);
  const float *y = static_cast<const float *>(inputs[1]);

  float *out = static_cast<float *>(outputs[0]);

  int num = prev_size * midd_size * post_size;
  int thread = 256;
  int block = (num + thread - 1) / thread;
  if (type_ == "add") {
    elementwise_kernel<<<block, thread, 0, stream>>>(
        num, x, y, out, prev_size, midd_size, post_size, details::Add<float>());
  } else if (type_ == "mul") {
    elementwise_kernel<<<block, thread, 0, stream>>>(
        num, x, y, out, prev_size, midd_size, post_size, details::Mul<float>());
  } else {
    PADDLE_THROW("Not implemented.");
  }

  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
