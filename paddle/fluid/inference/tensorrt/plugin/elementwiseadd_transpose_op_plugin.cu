/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/transpose_kernel.h"

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
      common::errors::InvalidArgument("The input of elementwiseadd_transpose "
                                      "plugin should not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      common::errors::InvalidArgument("The pos(%d) should be less than the "
                                      "num(%d) of the input and the output.",
                                      pos,
                                      nb_inputs + nb_outputs));
  // (in_out && pos < (nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  // input 0
  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kHALF ||
            in.type == nvinfer1::DataType::kFLOAT) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  // input 1
  if (pos == 1) {
    return (in.type == in_out[0].type) &&
           (in.format == nvinfer1::TensorFormat::kLINEAR);
  }
  // output 0
  if (pos == 2) {
    // 7.0.0.11 test_pcpvt_base_trt_fp16.py failed if support C8.
    // Only support linear format in lower versions of TRT
#if IS_TRT_VERSION_GE(7100)
    bool support_format = in.format == nvinfer1::TensorFormat::kLINEAR ||
                          in.format == nvinfer1::TensorFormat::kHWC8;
#else
    bool support_format = in.format == nvinfer1::TensorFormat::kLINEAR;
#endif

    return (in.type == in_out[0].type) && (support_format);
  }
}
void ElementwiseAddTransposePluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *input_desc,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *output_desc,
    int nbOutputs) TRT_NOEXCEPT {
  const auto &x_dims = input_desc[0].desc.dims;
  const auto &y_dims = input_desc[1].desc.dims;
  const auto &out_dims = output_desc[0].desc.dims;
  const auto &x_type = input_desc[0].desc.type;
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
  x_numel_ = x_numel;
  y_numel_ = y_numel;
  out_numel_ = out_numel;
  if (x_numel <= 0) {
    return;
  }
  ele_out_tensor_.Resize(common::make_ddim(x_shape));
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  auto *device_context = static_cast<phi::GPUContext *>(pool.Get(place));
  const phi::GPUContext &dev_ctx = *device_context;

  if (x_type == nvinfer1::DataType::kFLOAT) {
    x_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                   common::make_ddim(x_shape));
    y_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                   common::make_ddim(y_shape));
    out_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT32,
                                     common::make_ddim(out_shape));
    dev_ctx.template Alloc<float>(&ele_out_tensor_, x_numel * sizeof(float));
  } else if (x_type == nvinfer1::DataType::kHALF) {
    x_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT16,
                                   common::make_ddim(x_shape));
    y_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT16,
                                   common::make_ddim(y_shape));
    out_meta_ = phi::DenseTensorMeta(phi::DataType::FLOAT16,
                                     common::make_ddim(out_shape));
    dev_ctx.template Alloc<phi::dtype::float16>(
        &ele_out_tensor_, x_numel * sizeof(phi::dtype::float16));
  }
}
nvinfer1::DataType ElementwiseAddTransposePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index,
                    0,
                    common::errors::InvalidArgument(
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
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  phi::GPUPlace place(platform::GetCurrentDeviceId());
  auto *device_context = static_cast<phi::GPUContext *>(pool.Get(place));
  const phi::GPUContext &dev_ctx = *device_context;

  auto input_type = input_desc[0].type;
  auto output_format = output_desc[0].format;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. elementwiseadd_transpose-->fp32";
    const float *x = static_cast<const float *>(inputs[0]);
    const float *y = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    VLOG(1) << "TRT Plugin format selected. elementwiseadd_transpose-->kLINEAR";
    std::shared_ptr<phi::Allocation> x_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(x)),  // NOLINT
        x_numel_ * sizeof(float),
        place));
    std::shared_ptr<phi::Allocation> y_alloc(new phi::Allocation(
        static_cast<void *>(const_cast<float *>(y)),  // NOLINT
        y_numel_ * sizeof(float),
        place));
    std::shared_ptr<phi::Allocation> out_alloc(
        new phi::Allocation(static_cast<void *>(out),  // NOLINT
                            out_numel_ * sizeof(float),
                            place));
    const phi::DenseTensor x_tensor = phi::DenseTensor(x_alloc, x_meta_);
    const phi::DenseTensor y_tensor = phi::DenseTensor(y_alloc, y_meta_);
    phi::DenseTensor out_tensor = phi::DenseTensor(out_alloc, out_meta_);
    phi::AddKernel<float, phi::GPUContext>(
        dev_ctx, x_tensor, y_tensor, &ele_out_tensor_);
    phi::TransposeKernel<float, phi::GPUContext>(
        dev_ctx, ele_out_tensor_, std::vector<int>{0, 2, 1}, &out_tensor);
  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. elementwiseadd_transpose-->fp16";
    const half *x = static_cast<const half *>(inputs[0]);
    const half *y = static_cast<const half *>(inputs[1]);
    half *out = static_cast<half *>(outputs[0]);
    if (output_format == nvinfer1::PluginFormat::kLINEAR) {
      VLOG(1)
          << "TRT Plugin format selected. elementwiseadd_transpose-->kLINEAR";
      std::shared_ptr<phi::Allocation> x_alloc(new phi::Allocation(
          static_cast<void *>(const_cast<half *>(x)),  // NOLINT
          x_numel_ * sizeof(half),
          place));
      std::shared_ptr<phi::Allocation> y_alloc(new phi::Allocation(
          static_cast<void *>(const_cast<half *>(y)),  // NOLINT
          y_numel_ * sizeof(half),
          place));

      std::shared_ptr<phi::Allocation> out_alloc(
          new phi::Allocation(static_cast<void *>(out),  // NOLINT
                              out_numel_ * sizeof(half),
                              place));
      const phi::DenseTensor x_tensor = phi::DenseTensor(x_alloc, x_meta_);
      const phi::DenseTensor y_tensor = phi::DenseTensor(y_alloc, y_meta_);
      phi::DenseTensor out_tensor = phi::DenseTensor(out_alloc, out_meta_);
      phi::AddKernel<phi::dtype::float16, phi::GPUContext>(
          dev_ctx, x_tensor, y_tensor, &ele_out_tensor_);
      phi::TransposeKernel<phi::dtype::float16, phi::GPUContext>(
          dev_ctx, ele_out_tensor_, std::vector<int>{0, 2, 1}, &out_tensor);
    } else if (output_format == nvinfer1::PluginFormat::kHWC8) {
      VLOG(1) << "TRT Plugin format selected. elementwiseadd_transpose-->kHWC8";
      std::shared_ptr<phi::Allocation> x_alloc(new phi::Allocation(
          static_cast<void *>(const_cast<half *>(x)),  // NOLINT
          x_numel_ * sizeof(half),
          place));
      std::shared_ptr<phi::Allocation> y_alloc(new phi::Allocation(
          static_cast<void *>(const_cast<half *>(y)),  // NOLINT
          y_numel_ * sizeof(half),
          place));
      std::shared_ptr<phi::Allocation> out_alloc(
          new phi::Allocation(static_cast<void *>(out),  // NOLINT
                              out_numel_ * sizeof(half),
                              place));
      const phi::DenseTensor x_tensor = phi::DenseTensor(x_alloc, x_meta_);
      const phi::DenseTensor y_tensor = phi::DenseTensor(y_alloc, y_meta_);
      phi::DenseTensor out_tensor = phi::DenseTensor(out_alloc, out_meta_);
      phi::AddKernel<phi::dtype::float16, phi::GPUContext>(
          dev_ctx, x_tensor, y_tensor, &out_tensor);
    }
  }
  return cudaGetLastError() != cudaSuccess;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
