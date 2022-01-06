/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

#include "paddle/fluid/inference/tensorrt/plugin/deformable_conv_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

static inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                                 int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

nvinfer1::Weights DeformableConvPlugin::copyToDevice(const void* hostData,
                                                     size_t count) {
  int num_bytes = (data_type_ == nvinfer1::DataType::kFLOAT ? 4 : 2);
  void* deviceData;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&deviceData, count * num_bytes));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(deviceData, hostData, count * num_bytes,
                                        cudaMemcpyHostToDevice));
  return nvinfer1::Weights{data_type_, deviceData, int64_t(count)};
}

void DeformableConvPlugin::serializeFromDevice(
    void** hostBuffer, const nvinfer1::Weights& deviceWeights) const {
  int num_bytes = (data_type_ == nvinfer1::DataType::kFLOAT ? 4 : 2);
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemcpy(static_cast<char*>(*hostBuffer), deviceWeights.values,
                 deviceWeights.count * num_bytes, cudaMemcpyDeviceToHost));
  *hostBuffer =
      reinterpret_cast<char*>(*hostBuffer) + deviceWeights.count * num_bytes;
}

nvinfer1::Weights DeformableConvPlugin::deserializeToDevice(
    const void** hostBuffer, size_t count) {
  int num_bytes = (data_type_ == nvinfer1::DataType::kFLOAT ? 4 : 2);
  nvinfer1::Weights w =
      copyToDevice(static_cast<const char*>(*hostBuffer), count);
  *hostBuffer = reinterpret_cast<const char*>(*hostBuffer) + count * num_bytes;
  return w;
}

DeformableConvPlugin::DeformableConvPlugin(
    const nvinfer1::DataType data_type, const nvinfer1::Weights& weights,
    const std::vector<int>& kernel_dims, const std::vector<int>& strides,
    const std::vector<int>& paddings, const std::vector<int>& dilations,
    const int groups, const int deformable_groups, const int im2col_step,
    const bool with_fp16)
    : data_type_(data_type),
      groups_(groups),
      deformable_groups_(deformable_groups),
      im2col_step_(im2col_step),
      with_fp16_(with_fp16) {
  weights_ = copyToDevice(weights.values, weights.count);
  kernel_dims_.insert(kernel_dims_.end(), kernel_dims.cbegin(),
                      kernel_dims.cend());

  strides_.insert(strides_.end(), strides.cbegin(), strides.cend());
  paddings_.insert(paddings_.end(), paddings.cbegin(), paddings.cend());
  dilations_.insert(dilations_.end(), dilations.cbegin(), dilations.cend());
  PADDLE_ENFORCE_EQ(data_type_ == nvinfer1::DataType::kFLOAT ||
                        data_type_ == nvinfer1::DataType::kHALF,
                    true, platform::errors::InvalidArgument(
                              "The DeformableConv TRT Plugin's input type "
                              "should be float or half."));
  PADDLE_ENFORCE_EQ(
      paddings_.size(), strides_.size(),
      platform::errors::InvalidArgument(
          "The size of paddings (%d) is not equal to the size of strides (%d).",
          paddings_.size(), strides_.size()));
}

DeformableConvPlugin::DeformableConvPlugin(
    const nvinfer1::DataType data_type, const nvinfer1::Weights& weights,
    const std::vector<int>& kernel_dims, const std::vector<int>& strides,
    const std::vector<int>& paddings, const std::vector<int>& dilations,
    const int groups, const int deformable_groups, const int im2col_step,
    const std::vector<int>& input_dim, const std::vector<int>& offset_dim,
    const std::vector<int>& mask_dim, const std::vector<int>& output_dim,
    const bool with_fp16)
    : data_type_(data_type),
      groups_(groups),
      deformable_groups_(deformable_groups),
      im2col_step_(im2col_step),
      with_fp16_(with_fp16) {
  weights_ = copyToDevice(weights.values, weights.count);
  kernel_dims_.insert(kernel_dims_.end(), kernel_dims.cbegin(),
                      kernel_dims.cend());

  strides_.insert(strides_.end(), strides.cbegin(), strides.cend());
  paddings_.insert(paddings_.end(), paddings.cbegin(), paddings.cend());
  dilations_.insert(dilations_.end(), dilations.cbegin(), dilations.cend());
  input_dim_.insert(input_dim_.end(), input_dim.cbegin(), input_dim.cend());
  offset_dim_.insert(offset_dim_.end(), offset_dim.cbegin(), offset_dim.cend());
  mask_dim_.insert(mask_dim_.end(), mask_dim.cbegin(), mask_dim.cend());
  output_dim_.insert(output_dim_.end(), output_dim.cbegin(), output_dim.cend());
  PADDLE_ENFORCE_EQ(data_type_ == nvinfer1::DataType::kFLOAT ||
                        data_type_ == nvinfer1::DataType::kHALF,
                    true, platform::errors::InvalidArgument(
                              "The DeformableConv TRT Plugin's input type "
                              "should be float or half."));
  PADDLE_ENFORCE_EQ(
      paddings_.size(), strides_.size(),
      platform::errors::InvalidArgument(
          "The size of paddings (%d) is not equal to the size of strides (%d).",
          paddings_.size(), strides_.size()));
}

DeformableConvPlugin::DeformableConvPlugin(const void* data, size_t length) {
  DeserializeValue(&data, &length, &data_type_);
  DeserializeValue(&data, &length, &strides_);
  DeserializeValue(&data, &length, &paddings_);
  DeserializeValue(&data, &length, &dilations_);
  DeserializeValue(&data, &length, &groups_);
  DeserializeValue(&data, &length, &deformable_groups_);
  DeserializeValue(&data, &length, &im2col_step_);
  DeserializeValue(&data, &length, &kernel_dims_);
  int64_t count;
  DeserializeValue(&data, &length, &count);
  weights_ = deserializeToDevice(&data, count);
  DeserializeValue(&data, &length, &input_dim_);
  DeserializeValue(&data, &length, &offset_dim_);
  DeserializeValue(&data, &length, &mask_dim_);
  DeserializeValue(&data, &length, &output_dim_);
  DeserializeValue(&data, &length, &with_fp16_);
}

DeformableConvPlugin::~DeformableConvPlugin() {
  if (weights_.values) {
    cudaFree(const_cast<void*>(weights_.values));
    weights_.values = nullptr;
  }
}

const char* DeformableConvPlugin::getPluginType() const TRT_NOEXCEPT {
  return "deformable_conv_plugin";
}

const char* DeformableConvPlugin::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

int DeformableConvPlugin::getNbOutputs() const TRT_NOEXCEPT { return 1; }

nvinfer1::Dims DeformableConvPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nb_input_dims) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(nb_input_dims, 3,
                    platform::errors::InvalidArgument(
                        "The number of inputs should be equal to 3, but got %d",
                        nb_input_dims));
  nvinfer1::Dims ret;
  ret.nbDims = inputs[0].nbDims;
  ret.d[0] = kernel_dims_[0];
  ret.d[1] = ConvOutputSize(inputs[0].d[1], kernel_dims_[2], dilations_[0],
                            paddings_[0], strides_[0]);
  ret.d[2] = ConvOutputSize(inputs[0].d[2], kernel_dims_[3], dilations_[1],
                            paddings_[1], strides_[1]);
  return ret;
}

bool DeformableConvPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::TensorFormat format) const TRT_NOEXCEPT {
  if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    return (type == nvinfer1::DataType::kHALF) &&
           (format == nvinfer1::TensorFormat::kLINEAR);
#else
    return (type == nvinfer1::DataType::kFLOAT) &&
           (format == nvinfer1::TensorFormat::kLINEAR);
#endif
  } else {
    return (type == nvinfer1::DataType::kFLOAT) &&
           (format == nvinfer1::TensorFormat::kLINEAR);
  }
}

size_t DeformableConvPlugin::getWorkspaceSize(int max_batch_size) const
    TRT_NOEXCEPT {
  int c_i = input_dim_[0], h_i = input_dim_[1], w_i = input_dim_[2];
  int k_h = kernel_dims_[2], k_w = kernel_dims_[3];
  int c_o = output_dim_[0], h_o = output_dim_[1], w_o = output_dim_[2];
  int num_bytes = (data_type_ == nvinfer1::DataType::kFLOAT ? 4 : 2);
  size_t data_col_size = static_cast<size_t>(c_i * k_h * k_w * im2col_step_ *
                                             h_o * w_o * num_bytes);
  return data_col_size;
}

int DeformableConvPlugin::enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
                                  void** outputs, void* workspace,
#else
                                  void* const* outputs, void* workspace,
#endif
                                  cudaStream_t stream) TRT_NOEXCEPT {
  if (data_type_ == nvinfer1::DataType::kFLOAT) {
    enqueue_impl<float>(batch_size, inputs, outputs, workspace, stream);
  } else if (data_type_ == nvinfer1::DataType::kHALF) {
#if TRT_PLUGIN_FP16_AVALIABLE
    enqueue_impl<half>(batch_size, inputs, outputs, workspace, stream);
#else
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Current CUDA arch dose not support fp16. Please use fp32 instead."));
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The DeformableConv TRT Plugin's input type should be float or half."));
  }
  return cudaGetLastError() != cudaSuccess;
}

template <typename T>
__device__ T kFloor(T x);

template <>
__device__ half kFloor<half>(half x) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  return hfloor(x);
#endif
}

template <>
__device__ float kFloor<float>(float x) {
  return floor(x);
}

template <typename T>
__device__ T DmcnIm2colBilinear(const T* bottom_data, const int data_width,
                                const int height, const int width, T h, T w);

template <>
__device__ float DmcnIm2colBilinear<float>(const float* bottom_data,
                                           const int data_width,
                                           const int height, const int width,
                                           float h, float w) {
  int h_low = kFloor<float>(h);
  int w_low = kFloor<float>(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float h_low_t = h_low, w_low_t = w_low, one = 1.0f;
  float lh = h - h_low_t;
  float lw = w - w_low_t;
  float hh = one - lh, hw = one - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ half DmcnIm2colBilinear<half>(const half* bottom_data,
                                         const int data_width, const int height,
                                         const int width, half h, half w) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int h_low = kFloor<half>(h);
  int w_low = kFloor<half>(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  half h_low_t = h_low, w_low_t = w_low, one = 1.0f;
  half lh = h - h_low_t;
  half lw = w - w_low_t;
  half hh = one - lh, hw = one - lw;

  half v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
  half v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  half v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  half v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  half w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  half val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
#endif
}

template <typename T>
__global__ void ModulatedDeformableIm2colGpuKernel(
    const int nthreads, const T* data_im, const T* data_offset,
    const T* data_mask, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T* data_col);

template <>
__global__ void ModulatedDeformableIm2colGpuKernel<float>(
    const int nthreads, const float* data_im, const float* data_offset,
    const float* data_mask, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col, float* data_col) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;

  float minus_one = -1.0f, height_t = height, width_t = width;
  for (size_t i = index; i < nthreads; i += offset) {
    const int w_col = i % width_col;
    const int h_col = (i / width_col) % height_col;
    const int b_col = (i / width_col) / height_col % batch_size;
    const int c_im = (i / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    float* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const float* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const float* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = 0;
        float h_im_t = h_in + i * dilation_h, w_im_t = w_in + j * dilation_w;
        const float h_im = h_im_t + offset_h;
        const float w_im = w_im_t + offset_w;
        if (h_im > minus_one && w_im > minus_one && h_im < height_t &&
            w_im < width_t) {
          val = DmcnIm2colBilinear<float>(data_im_ptr, width, height, width,
                                          h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <>
__global__ void ModulatedDeformableIm2colGpuKernel<half>(
    const int nthreads, const half* data_im, const half* data_offset,
    const half* data_mask, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col, half* data_col) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;

  half minus_one = -1.0f, height_t = height, width_t = width;
  for (size_t i = index; i < nthreads; i += offset) {
    const int w_col = i % width_col;
    const int h_col = (i / width_col) % height_col;
    const int b_col = (i / width_col) / height_col % batch_size;
    const int c_im = (i / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    half* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const half* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const half* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const half* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const half offset_h = data_offset_ptr[data_offset_h_ptr];
        const half offset_w = data_offset_ptr[data_offset_w_ptr];
        const half mask = data_mask_ptr[data_mask_hw_ptr];
        half val = 0;
        half h_im_t = h_in + i * dilation_h, w_im_t = w_in + j * dilation_w;
        const half h_im = h_im_t + offset_h;
        const half w_im = w_im_t + offset_w;
        if (h_im > minus_one && w_im > minus_one && h_im < height_t &&
            w_im < width_t) {
          val = DmcnIm2colBilinear<half>(data_im_ptr, width, height, width,
                                         h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
#endif
}

template <typename T>
void gemm_impl(cublasHandle_t handle, cublasOperation_t transa,
               cublasOperation_t transb, int m, int n, int k, const T* alpha,
               const T* A, int lda, const T* B, int ldb, const T* beta, T* C,
               int ldc);

template <>
void gemm_impl<float>(cublasHandle_t handle, cublasOperation_t transa,
                      cublasOperation_t transb, int m, int n, int k,
                      const float* alpha, const float* A, int lda,
                      const float* B, int ldb, const float* beta, float* C,
                      int ldc) {
  platform::dynload::cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                 B, ldb, beta, C, ldc);
}

template <>
void gemm_impl<half>(cublasHandle_t handle, cublasOperation_t transa,
                     cublasOperation_t transb, int m, int n, int k,
                     const half* alpha, const half* A, int lda, const half* B,
                     int ldb, const half* beta, half* C, int ldc) {
#if TRT_PLUGIN_FP16_AVALIABLE
  platform::dynload::cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                 B, ldb, beta, C, ldc);
#else
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Current CUDA arch dose not support fp16. Please use fp32 instead."));
#endif
}

template <typename T>
int DeformableConvPlugin::enqueue_impl(int batch_size,
                                       const void* const* inputs,
                                       void* const* outputs, void* workspace,
                                       cudaStream_t stream) {
  const T* input = reinterpret_cast<const T*>(inputs[0]);
  const T* offset = reinterpret_cast<const T*>(inputs[1]);
  const T* mask = reinterpret_cast<const T*>(inputs[2]);
  const T* filter = reinterpret_cast<const T*>(weights_.values);
  T* output = reinterpret_cast<T*>(outputs[0]);

  int c_i = input_dim_[0], h_i = input_dim_[1], w_i = input_dim_[2];
  int k_h = kernel_dims_[2], k_w = kernel_dims_[3];
  int c_o = output_dim_[0], h_o = output_dim_[1], w_o = output_dim_[2];

  int input_stride = c_i * h_i * w_i;
  int offset_stride = offset_dim_[0] * offset_dim_[1] * offset_dim_[2];
  int mask_stride = mask_dim_[0] * mask_dim_[1] * mask_dim_[2];
  int output_stride = c_o * h_o * w_o;

  int M = c_o / groups_;
  int N = im2col_step_ * h_o * w_o;
  int K = c_i * k_h * k_w / groups_;

  // c_i / deformable_groups
  int channel_per_deformable_group = c_i / deformable_groups_;
  // c_i * im2col_step * h_o * w_o
  int num_kernels = c_i * im2col_step_ * h_o * w_o;

  int blocks = NumBlocks(num_kernels);
  int threads = kNumCUDAThreads;

  T alpha = static_cast<T>(1.0f);
  T beta = static_cast<T>(0.0f);

  for (int i = 0; i < batch_size / im2col_step_; ++i) {
    const T* data_im = input + i * im2col_step_ * input_stride;
    const T* data_offset = offset + i * im2col_step_ * offset_stride;
    const T* data_mask = mask + i * im2col_step_ * mask_stride;
    T* data_col = reinterpret_cast<T*>(workspace);

    ModulatedDeformableIm2colGpuKernel<T><<<blocks, threads, 0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, h_i, w_i, k_h, k_w,
        paddings_[0], paddings_[1], strides_[0], strides_[1], dilations_[0],
        dilations_[1], channel_per_deformable_group, im2col_step_, c_i,
        deformable_groups_, h_o, w_o, data_col);

    for (int g = 0; g < groups_; ++g) {
      const T* weight = filter + g * M * K;
      const T* col = data_col + g * K * N;
      T* out = output + i * im2col_step_ * output_stride + g * M * N;
      gemm_impl<T>(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                   col, N, weight, K, &beta, out, N);
    }
  }
  return 0;
}

int DeformableConvPlugin::initialize() TRT_NOEXCEPT { return 0; }

void DeformableConvPlugin::terminate() TRT_NOEXCEPT {}

size_t DeformableConvPlugin::getSerializationSize() const TRT_NOEXCEPT {
  size_t serialize_size = 0;
  serialize_size += SerializedSize(data_type_);
  serialize_size += SerializedSize(strides_);
  serialize_size += SerializedSize(paddings_);
  serialize_size += SerializedSize(dilations_);
  serialize_size += SerializedSize(groups_);
  serialize_size += SerializedSize(deformable_groups_);
  serialize_size += SerializedSize(im2col_step_);
  serialize_size += SerializedSize(kernel_dims_);
  serialize_size += SerializedSize(weights_.count);
  int num_bytes = (data_type_ == nvinfer1::DataType::kFLOAT ? 4 : 2);
  serialize_size += weights_.count * num_bytes;
  serialize_size += SerializedSize(input_dim_);
  serialize_size += SerializedSize(offset_dim_);
  serialize_size += SerializedSize(mask_dim_);
  serialize_size += SerializedSize(output_dim_);
  serialize_size += SerializedSize(with_fp16_);
  return serialize_size;
}

void DeformableConvPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, strides_);
  SerializeValue(&buffer, paddings_);
  SerializeValue(&buffer, dilations_);
  SerializeValue(&buffer, groups_);
  SerializeValue(&buffer, deformable_groups_);
  SerializeValue(&buffer, im2col_step_);
  SerializeValue(&buffer, kernel_dims_);
  SerializeValue(&buffer, weights_.count);
  serializeFromDevice(&buffer, weights_);
  SerializeValue(&buffer, input_dim_);
  SerializeValue(&buffer, offset_dim_);
  SerializeValue(&buffer, mask_dim_);
  SerializeValue(&buffer, output_dim_);
  SerializeValue(&buffer, with_fp16_);
}

void DeformableConvPlugin::destroy() TRT_NOEXCEPT {}

void DeformableConvPlugin::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  namespace_ = std::string(lib_namespace);
}

const char* DeformableConvPlugin::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_.c_str();
}

nvinfer1::DataType DeformableConvPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* input_type,
    int nb_inputs) const TRT_NOEXCEPT {
  return input_type[0];
}

bool DeformableConvPlugin::isOutputBroadcastAcrossBatch(
    int output_index, const bool* input_is_broadcast,
    int nb_inputs) const TRT_NOEXCEPT {
  return false;
}

bool DeformableConvPlugin::canBroadcastInputAcrossBatch(int input_index) const
    TRT_NOEXCEPT {
  return false;
}

void DeformableConvPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext,
    nvinfer1::IGpuAllocator* gpuAllocator) TRT_NOEXCEPT {
  cublasHandle_ = cublasContext;
}

void DeformableConvPlugin::configurePlugin(
    const nvinfer1::Dims* input_dims, int nb_inputs,
    const nvinfer1::Dims* output_dims, int nb_outputs,
    const nvinfer1::DataType* input_types,
    const nvinfer1::DataType* output_types, const bool* input_is_broadcast,
    const bool* output_is_broadcast, nvinfer1::PluginFormat float_format,
    int max_batct_size) TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(
      nb_inputs, 3,
      platform::errors::InvalidArgument(
          "The number of inputs should be equal to 3, but got %d", nb_inputs));
  PADDLE_ENFORCE_EQ(
      nb_outputs, 1,
      platform::errors::InvalidArgument(
          "The number of inputs should be equal to 1, but got %d", nb_outputs));

  for (int i = 0; i < input_dims[0].nbDims; i++) {
    input_dim_.push_back(input_dims[0].d[i]);
  }
  for (int i = 0; i < input_dims[1].nbDims; i++) {
    offset_dim_.push_back(input_dims[1].d[i]);
  }
  for (int i = 0; i < input_dims[2].nbDims; i++) {
    mask_dim_.push_back(input_dims[2].d[i]);
  }
  for (int i = 0; i < output_dims[0].nbDims; i++) {
    output_dim_.push_back(output_dims[0].d[i]);
  }
}

nvinfer1::IPluginV2Ext* DeformableConvPlugin::clone() const TRT_NOEXCEPT {
  return new DeformableConvPlugin(
      data_type_, weights_, kernel_dims_, strides_, paddings_, dilations_,
      groups_, deformable_groups_, im2col_step_, input_dim_, offset_dim_,
      mask_dim_, output_dim_, with_fp16_);
}

void DeformableConvPluginCreator::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  namespace_ = std::string(lib_namespace);
}

const char* DeformableConvPluginCreator::getPluginNamespace() const
    TRT_NOEXCEPT {
  return namespace_.c_str();
}

const char* DeformableConvPluginCreator::getPluginName() const TRT_NOEXCEPT {
  return "deformable_conv_plugin";
}

const char* DeformableConvPluginCreator::getPluginVersion() const TRT_NOEXCEPT {
  return "1";
}

const nvinfer1::PluginFieldCollection*
DeformableConvPluginCreator::getFieldNames() TRT_NOEXCEPT {
  return &field_collection_;
}

nvinfer1::IPluginV2Ext* DeformableConvPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  const nvinfer1::PluginField* fields = fc->fields;

  nvinfer1::DataType data_type;
  std::vector<int> strides, paddings, dilations, kernel_dims;
  nvinfer1::Weights weights;
  int groups = -1;
  int deformable_groups = -1;
  int im2col_step = -1;
  bool with_fp16 = false;

  for (int i = 0; i < fc->nbFields; ++i) {
    const std::string field_name(fc->fields[i].name);
    if (field_name.compare("data_type") == 0) {
      data_type = *static_cast<const nvinfer1::DataType*>(fc->fields[i].data);
    } else if (field_name.compare("strides")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      strides.insert(strides.end(), data, data + length);
    } else if (field_name.compare("paddings")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      paddings.insert(paddings.end(), data, data + length);
    } else if (field_name.compare("dilations")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      dilations.insert(dilations.end(), data, data + length);
    } else if (field_name.compare("groups")) {
      groups = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("deformable_groups")) {
      deformable_groups = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("im2col_step")) {
      im2col_step = *static_cast<const int*>(fc->fields[i].data);
    } else if (field_name.compare("kernel_dims")) {
      const int length = fc->fields[i].length;
      const int* data = static_cast<const int*>(fc->fields[i].data);
      kernel_dims.insert(kernel_dims.end(), data, data + length);
    } else if (field_name.compare("weights")) {
      weights.count = fc->fields[i].length;
      weights.values = fc->fields[i].data;
    } else if (field_name.compare("with_fp16")) {
      with_fp16 = *static_cast<const bool*>(fc->fields[i].data);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unknown plugin field name [%s] in the DeformableConv TRT Plugin.",
          field_name));
    }
  }
  weights.type = data_type;
  return new DeformableConvPlugin(data_type, weights, kernel_dims, strides,
                                  paddings, dilations, groups,
                                  deformable_groups, im2col_step, with_fp16);
}

nvinfer1::IPluginV2Ext* DeformableConvPluginCreator::deserializePlugin(
    const char* name, const void* serial_data,
    size_t serial_length) TRT_NOEXCEPT {
  auto plugin = new DeformableConvPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
