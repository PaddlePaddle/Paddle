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
#include "paddle/fluid/inference/tensorrt/plugin/elementwiseadd_transpose_op_plugin.h"
#include <glog/logging.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

int ElementwiseAddTransposePluginDynamic::initialize() TRT_NOEXCEPT {
  return 0;
}

size_t ElementwiseAddTransposePluginDynamic::getSerializationSize() const
    TRT_NOEXCEPT {
  return SerializedSize(axis_) + SerializedSize(output_shape_);
}

void ElementwiseAddTransposePluginDynamic::serialize(void *buffer) const
    TRT_NOEXCEPT {
  SerializeValue(&buffer, axis_);
  SerializeValue(&buffer, output_shape_);
}

nvinfer1::DimsExprs ElementwiseAddTransposePluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs *inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[2];
  ret.d[2] = expr_builder.constant(output_shape_[1]);
  ret.d[3] = expr_builder.constant(output_shape_[2]);
  return ret;
}

bool ElementwiseAddTransposePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc *in_out,
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
  (in_out && pos < (nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  if (pos == 1) {
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  if (pos == 2) {
    return (in.type == nvinfer1::DataType::kFLOAT ||
            in.type == nvinfer1::DataType::kHALF) &&
           (in.format == nvinfer1::TensorFormat::kHWC8);
  }
}

nvinfer1::DataType ElementwiseAddTransposePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    platform::errors::InvalidArgument(
                        "The Elementwise Plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  return input_types[0];
}

int ElementwiseAddTransposePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc,
    const void *const *inputs,
    void *const *outputs,
    void *workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto x_dims = input_desc[0].dims;
  auto y_dims = input_desc[1].dims;
  auto out_dims = output_desc[0].dims;
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
    PADDLE_ENFORCE_EQ(x_dims.d[i + axis],
                      y_dims.d[i],
                      platform::errors::InvalidArgument(
                          "Broadcast dimension mismatch found in trt "
                          "elementwise plugin's x and y input."));
    midd_size *= y_dims.d[i];
  }

  for (int i = axis + trimed_nb_dims; i < x_dims.nbDims; ++i) {
    post_size *= x_dims.d[i];
  }
  auto input_type = input_desc[0].type;

  std::vector<int> x_shape;
  int x_numel = 1;
  for (int i = 0; i < x_dims.nbDims; i++) {
    x_shape.push_back(x_dims.d[i]);
    x_numel *= x_dims.d[i];
  }
  std::vector<int> y_shape;
  int y_numel = 1;
  for (int i = 0; i < y_dims.nbDims; i++) {
    y_shape.push_back(y_dims.d[i]);
    y_numel *= y_dims.d[i];
  }
  std::vector<int> out_shape;
  int out_numel = 1;
  for (int i = 0; i < out_dims.nbDims; i++) {
    out_shape.push_back(out_dims.d[i]);
    out_numel *= out_dims.d[i];
  }

  paddle::platform::DeviceContextPool &pool =
      paddle::platform::DeviceContextPool::Instance();
  platform::CUDAPlace place(platform::GetCurrentDeviceId());
  auto *device_context = static_cast<phi::GPUContext *>(pool.Get(place));
  const phi::GPUContext &dev_ctx = *device_context;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    const float *x = static_cast<const float *>(inputs[0]);
    const float *y = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

  } else if (input_type == nvinfer1::DataType::kHALF) {
    const half *x = static_cast<const half *>(inputs[0]);
    const half *y = static_cast<const half *>(inputs[1]);
    half *out = static_cast<half *>(outputs[0]);
    phi::DenseTensorMeta x_meta(phi::DataType::FLOAT16,
                                phi::make_ddim(x_shape));
    phi::DenseTensorMeta y_meta(phi::DataType::FLOAT16,
                                phi::make_ddim(y_shape));
    phi::DenseTensorMeta out_meta(phi::DataType::FLOAT16,
                                  phi::make_ddim(out_shape));
    std::shared_ptr<phi::Allocation> x_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<half *>(x)),  // NOLINT
        x_numel * sizeof(half),
        place));
    std::shared_ptr<phi::Allocation> y_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<half *>(y)),  // NOLINT
        y_numel * sizeof(half),
        place));
    std::shared_ptr<phi::Allocation> out_alloc(
        new phi::Allocation(static_cast<void *>(out),  // NOLINT
                            out_numel * sizeof(half),
                            place));
    const phi::DenseTensor x_tensor = phi::DenseTensor(x_alloc, x_meta);
    const phi::DenseTensor y_tensor = phi::DenseTensor(y_alloc, y_meta);
    phi::DenseTensor out_tensor = phi::DenseTensor(out_alloc, out_meta);
    phi::AddKernel<phi::dtype::float16, phi::GPUContext>(
        dev_ctx, x_tensor, y_tensor, &out_tensor);
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
