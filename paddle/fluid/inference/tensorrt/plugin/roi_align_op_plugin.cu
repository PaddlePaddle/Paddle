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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "paddle/fluid/inference/tensorrt/plugin/roi_align_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <class T>
__inline__ __device__ T BilinearInterpolate(const T* input_data,
                                            const int height, const int width,
                                            T y, T x) {
  if (y < -1.f || y > height || x < -1.f || x > width) return 0;
  y = y <= 0.f ? 0.f : y;
  x = x <= 0.f ? 0.f : x;
  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<T>(y_low);
  } else {
    y_high = y_low + 1;
  }
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<T>(x_low);
  } else {
    x_high = x_low + 1;
  }
  T ly = y - y_low, lx = x - x_low;
  T hy = 1.f - ly, hx = 1.f - lx;
  T v1 = input_data[y_low * width + x_low];
  T v2 = input_data[y_low * width + x_high];
  T v3 = input_data[y_high * width + x_low];
  T v4 = input_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T, typename OutT, bool USE_SMEM>
__global__ void GPUROIAlignOpt(const int nthreads,
                               const T* __restrict__ input_data,
                               const T* __restrict__ input_rois,
                               const float spatial_scale, const int channels,
                               const int height, const int width,
                               const int pooled_height, const int pooled_width,
                               const int sampling_ratio, const int num_rois,
                               OutT* __restrict__ output_data) {
  const int batch = blockIdx.x;
  const int channel = blockIdx.y;
  const T* offset_input_data =
      input_data + (batch * channels + channel) * height * width;
  extern __shared__ T s_input_data[];
  if (USE_SMEM) {
    for (int idx = threadIdx.x; idx < height * width; idx += blockDim.x) {
      s_input_data[idx] = offset_input_data[idx];
    }
    __syncthreads();
  }
  for (int idx = threadIdx.x; idx < num_rois * pooled_height * pooled_width;
       idx += blockDim.x) {
    const int pw = idx % pooled_width;
    const int ph = (idx / pooled_width) % pooled_height;
    const int roi_idx = (idx / pooled_width / pooled_height) % num_rois;
    const int n = batch * num_rois + roi_idx;
    const float4 rois_offset = reinterpret_cast<const float4*>(input_rois)[n];
    const T roi_xmin = rois_offset.x * spatial_scale;
    const T roi_ymin = rois_offset.y * spatial_scale;
    const T roi_xmax = rois_offset.z * spatial_scale;
    const T roi_ymax = rois_offset.w * spatial_scale;
    const T roi_width = max(roi_xmax - roi_xmin, static_cast<T>(1.f));
    const T roi_height = max(roi_ymax - roi_ymin, static_cast<T>(1.f));
    const T bin_size_h = roi_height / static_cast<T>(pooled_height);
    const T bin_size_w = roi_width / static_cast<T>(pooled_width);
    const int roi_bin_grid_h = (sampling_ratio > 0)
                                   ? sampling_ratio
                                   : ceil(roi_height / pooled_height);
    const int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    const T count = roi_bin_grid_h * roi_bin_grid_w;

    T output_val = 0.f;
    for (int iy = 0; iy < roi_bin_grid_h; ++iy) {
      const T y = roi_ymin + ph * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ++ix) {
        const T x = roi_xmin + pw * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);
        if (USE_SMEM) {
          T val = BilinearInterpolate<T>(s_input_data, height, width, y, x);
          output_val += val;
        } else {
          T val =
              BilinearInterpolate<T>(offset_input_data, height, width, y, x);
          output_val += val;
        }
      }
    }
    output_val /= count;
    const int out_offset =
        batch * num_rois * channels * pooled_height * pooled_width +
        roi_idx * channels * pooled_height * pooled_width +
        channel * pooled_height * pooled_width + ph * pooled_width + pw;
    output_data[out_offset] = static_cast<OutT>(output_val);
  }
}

#if IS_TRT_VERSION_GE(6000)
RoiAlignPluginDynamic::RoiAlignPluginDynamic(const nvinfer1::DataType data_type,
                                             const int pooled_height,
                                             const int pooled_width,
                                             float spatial_scale,
                                             int sampling_ratio)
    : data_type_(data_type),
      pooled_height_(pooled_height),
      pooled_width_(pooled_width),
      spatial_scale_(spatial_scale),
      sampling_ratio_(sampling_ratio) {
  bool data_type_is_valid = data_type_ == nvinfer1::DataType::kFLOAT ||
                            data_type_ == nvinfer1::DataType::kHALF;
  PADDLE_ENFORCE_EQ(data_type_is_valid, true,
                    platform::errors::InvalidArgument(
                        "TRT RoiAlign plugin only accepts kFLOAT(%d) or "
                        "kHALF(%d) data type, but the received data type = %d",
                        static_cast<int>(nvinfer1::DataType::kFLOAT),
                        static_cast<int>(nvinfer1::DataType::kHALF),
                        static_cast<int>(data_type_)));

  PADDLE_ENFORCE_GT(pooled_height_, 0,
                    platform::errors::InvalidArgument(
                        "TRT RoiAlign plugin only accepts pooled_height "
                        "greater than %d, but the received pooled_height = %d",
                        0, pooled_height_));

  PADDLE_ENFORCE_GT(pooled_width_, 0,
                    platform::errors::InvalidArgument(
                        "TRT RoiAlign plugin only accepts pooled_width greater "
                        "than %d, but the received pooled_width = %d",
                        0, pooled_height_));

  PADDLE_ENFORCE_GT(spatial_scale_, 0.f,
                    platform::errors::InvalidArgument(
                        "TRT RoiAlign plugin only accepts spatial_scale "
                        "greater than %f, but the received spatial_scale = %f",
                        0, spatial_scale_));

  int smem_per_block = -1;
  int device = -1;
  cudaGetDevice(&device);

  PADDLE_ENFORCE_GE(
      device, 0,
      platform::errors::InvalidArgument(
          "The cuda device ID should be greater than %d, but device ID is %d",
          0, device));

  cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                         device);
  smem_per_block_ = smem_per_block;
}

RoiAlignPluginDynamic::RoiAlignPluginDynamic(void const* data, size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &pooled_height_);
  DeserializeValue(&data, &length, &pooled_width_);
  DeserializeValue(&data, &length, &spatial_scale_);
  DeserializeValue(&data, &length, &sampling_ratio_);
  int smem_per_block = -1;
  int device = -1;
  cudaGetDevice(&device);
  PADDLE_ENFORCE_GE(
      device, 0,
      platform::errors::InvalidArgument(
          "The cuda device ID should be greater than %d, but device ID is %d",
          0, device));
  cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock,
                         device);
  smem_per_block_ = smem_per_block;
}

nvinfer1::IPluginV2DynamicExt* RoiAlignPluginDynamic::clone() const
    TRT_NOEXCEPT {
  auto* plugin =
      new RoiAlignPluginDynamic(data_type_, pooled_height_, pooled_width_,
                                spatial_scale_, sampling_ratio_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DimsExprs RoiAlignPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret{};
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];  // roi
  ret.d[1] = inputs[0].d[1];  // X
  ret.d[2] = exprBuilder.constant(pooled_height_);
  ret.d[3] = exprBuilder.constant(pooled_width_);
  return ret;
}

bool RoiAlignPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) TRT_NOEXCEPT {
  if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos < 2) {  // input
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
  }
  return inOut[pos].type == data_type_;
}

void RoiAlignPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) TRT_NOEXCEPT {}

size_t RoiAlignPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

template <typename T, typename OutT>
int RoiAlignPluginDynamic::enqueue_impl(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  auto in_dims = inputDesc[0].dims;
  auto rois_dims = inputDesc[1].dims;
  auto out_dims = outputDesc[0].dims;

  int rois_num = rois_dims.d[0];
  if (rois_num == 0) return cudaGetLastError() != cudaSuccess;

  int batch = in_dims.d[0];
  int channels = in_dims.d[1];
  int height = in_dims.d[2];
  int width = in_dims.d[3];

  int output_size =
      out_dims.d[0] * out_dims.d[1] * out_dims.d[2] * out_dims.d[3];

  const dim3 blocks(batch, channels);
  const int threads = 512;

  if (smem_per_block_ < width * height * sizeof(T)) {
    GPUROIAlignOpt<T, OutT, false><<<blocks, threads, 0, stream>>>(
        output_size, static_cast<const T*>(inputs[0]),
        static_cast<const T*>(inputs[1]), spatial_scale_, channels, height,
        width, pooled_height_, pooled_width_, sampling_ratio_, rois_num / batch,
        static_cast<OutT*>(outputs[0]));
  } else {
    GPUROIAlignOpt<
        T, OutT, true><<<blocks, threads, width * height * sizeof(T), stream>>>(
        output_size, static_cast<const T*>(inputs[0]),
        static_cast<const T*>(inputs[1]), spatial_scale_, channels, height,
        width, pooled_height_, pooled_width_, sampling_ratio_, rois_num / batch,
        static_cast<OutT*>(outputs[0]));
  }

  return cudaGetLastError() != cudaSuccess;
}

int RoiAlignPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                                   const nvinfer1::PluginTensorDesc* outputDesc,
                                   const void* const* inputs,
                                   void* const* outputs, void* workspace,
                                   cudaStream_t stream) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(outputDesc[0].type, data_type_,
                    platform::errors::InvalidArgument(
                        "TRT RoiAlignPluginDynamic expects outputDesc[0].type "
                        "equal to data_type_"));

  if (data_type_ == nvinfer1::DataType::kHALF) {
    return enqueue_impl<float, half>(inputDesc, outputDesc, inputs, outputs,
                                     workspace, stream);
  }
  return enqueue_impl<float, float>(inputDesc, outputDesc, inputs, outputs,
                                    workspace, stream);
}

nvinfer1::DataType RoiAlignPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes,
    int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

const char* RoiAlignPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return "roi_align_plugin_dynamic";
}

int RoiAlignPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

int RoiAlignPluginDynamic::initialize() TRT_NOEXCEPT { return 0; }

void RoiAlignPluginDynamic::terminate() TRT_NOEXCEPT {}

size_t RoiAlignPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(pooled_height_);
  serialize_size += SerializedSize(pooled_width_);
  serialize_size += SerializedSize(spatial_scale_);
  serialize_size += SerializedSize(sampling_ratio_);
  return serialize_size;
}

void RoiAlignPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, pooled_height_);
  SerializeValue(&buffer, pooled_width_);
  SerializeValue(&buffer, spatial_scale_);
  SerializeValue(&buffer, sampling_ratio_);
}

void RoiAlignPluginDynamic::destroy() TRT_NOEXCEPT {}

RoiAlignPluginDynamicCreator::RoiAlignPluginDynamicCreator() {}

void RoiAlignPluginDynamicCreator::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  namespace_ = std::string(lib_namespace);
}

const char* RoiAlignPluginDynamicCreator::getPluginNamespace() const
    TRT_NOEXCEPT {
  return namespace_.c_str();
}

const char* RoiAlignPluginDynamicCreator::getPluginName() const TRT_NOEXCEPT {
  return "roi_align_plugin_dynamic";
}

const char* RoiAlignPluginDynamicCreator::getPluginVersion() const
    TRT_NOEXCEPT {
  return "1";
}

const nvinfer1::PluginFieldCollection*
RoiAlignPluginDynamicCreator::getFieldNames() TRT_NOEXCEPT {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* RoiAlignPluginDynamicCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  const nvinfer1::PluginField* fields = fc->fields;
  return nullptr;
}

nvinfer1::IPluginV2Ext* RoiAlignPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  auto plugin = new RoiAlignPluginDynamic(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
