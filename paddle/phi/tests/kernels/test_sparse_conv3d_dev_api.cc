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

#include <gtest/gtest.h>
#include <memory>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"

namespace phi {
namespace tests {

std::vector<int> flatten(const std::vector<std::vector<int>>& in) {
  std::vector<int> out;
  if (in.size() == 0) return out;
  const int rows = in.size();
  const int cols = in[0].size();
  const int n = rows * cols;
  out.resize(n);
  for (int i = 0; i < rows; i++) {
    memcpy(&out[i * cols], in[i].data(), cols * sizeof(int));
  }
  return out;
}

template <typename T1, typename T2>
void cast(const std::vector<T1>& in, std::vector<T2>* out) {
  for (uint64_t i = 0; i < in.size(); i++) {
    (*out)[i] = static_cast<T2>(in[i]);
  }
}

template <typename T>
void TestConv3d(const std::vector<int>& indices,
                const std::vector<T>& features,
                const DDim& x_dims,
                const std::vector<T>& kernel,
                const DDim& kernel_dims,
                const std::vector<int>& correct_out_indices,
                const std::vector<T>& correct_out_features,
                const DDim& correct_out_dims,
                const int non_zero_num,
                const std::vector<int>& paddings,
                const std::vector<int>& strides,
                const std::vector<int>& dilations) {
  const auto alloc = std::make_shared<paddle::experimental::DefaultAllocator>(
      paddle::platform::CPUPlace());

  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.Init();
  phi::CPUPlace cpu;

  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  DenseTensor indices_tensor(
      alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, non_zero_num}, DataLayout::NCHW));
  memcpy(indices_tensor.mutable_data<int>(cpu),
         indices.data(),
         indices.size() * sizeof(int));
  DenseTensor features_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NHWC));
  memcpy(features_tensor.mutable_data<T>(cpu),
         features.data(),
         features.size() * sizeof(T));

  SparseCooTensor x_tensor(indices_tensor, features_tensor, x_dims);

  // TODO(zhangkaihuo) change layout to DHWCOC
  DenseTensor kernel_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      kernel_dims,
                      DataLayout::NHWC));
  memcpy(kernel_tensor.mutable_data<T>(cpu),
         kernel.data(),
         kernel.size() * sizeof(T));

  const float diff = 1e-3;
  if (!std::is_same<T, phi::dtype::float16>::value) {
    SparseCooTensor out = sparse::Conv3d<T>(
        dev_ctx_cpu, x_tensor, kernel_tensor, paddings, dilations, strides, 1);

    ASSERT_EQ(correct_out_dims.size(), out.dims().size());
    for (int i = 0; i < correct_out_dims.size(); i++) {
      ASSERT_EQ(correct_out_dims[i], out.dims()[i]);
    }
    ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, out.nnz());

    int cmp_indices = memcmp(correct_out_indices.data(),
                             out.non_zero_indices().data<int>(),
                             correct_out_indices.size() * sizeof(int));
    ASSERT_EQ(cmp_indices, 0);

    for (uint64_t i = 0; i < correct_out_features.size(); i++) {
      float tmp = std::fabs(static_cast<float>(
          correct_out_features[i] - out.non_zero_elements().data<T>()[i]));
      ASSERT_LT(tmp, diff);
    }
  }

// test gpu
#if defined(PADDLE_WITH_CUDA)
  phi::GPUContext dev_ctx_gpu;
  dev_ctx_gpu.PartialInitWithoutAllocator();
  dev_ctx_gpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(dev_ctx_gpu.GetPlace(), dev_ctx_gpu.stream())
          .get());
  dev_ctx_gpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();

  const auto cuda_alloc =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          paddle::platform::CUDAPlace());
  DenseTensor d_indices_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, non_zero_num}, DataLayout::NCHW));
  phi::Copy(dev_ctx_gpu, indices_tensor, true, &d_indices_tensor);
  DenseTensor d_features_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NHWC));
  phi::Copy(dev_ctx_gpu, features_tensor, true, &d_features_tensor);

  SparseCooTensor d_x_tensor(d_indices_tensor, d_features_tensor, x_dims);

  // TODO(zhangkaihuo) change layout to DHWCOC
  DenseTensor d_kernel_tensor(
      cuda_alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      kernel_dims,
                      DataLayout::NHWC));
  phi::Copy(dev_ctx_gpu, kernel_tensor, true, &d_kernel_tensor);

  SparseCooTensor d_out = sparse::Conv3d<T>(dev_ctx_gpu,
                                            d_x_tensor,
                                            d_kernel_tensor,
                                            paddings,
                                            dilations,
                                            strides,
                                            1);

  ASSERT_EQ(correct_out_dims.size(), d_out.dims().size());
  ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, d_out.nnz());
  for (int i = 0; i < correct_out_dims.size(); i++) {
    ASSERT_EQ(correct_out_dims[i], d_out.dims()[i]);
  }

  DenseTensor h_indices_tensor(
      alloc.get(),
      DenseTensorMeta(DataType::INT32, {4, d_out.nnz()}, DataLayout::NCHW));
  phi::Copy(dev_ctx_gpu, d_out.non_zero_indices(), true, &h_indices_tensor);

  int cmp_indices2 = memcmp(correct_out_indices.data(),
                            h_indices_tensor.data<int>(),
                            correct_out_indices.size() * sizeof(int));
  ASSERT_EQ(cmp_indices2, 0);

  DenseTensor h_features_tensor(
      alloc.get(),
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {d_out.nnz()},
                      d_out.layout()));
  phi::Copy(dev_ctx_gpu, d_out.non_zero_elements(), true, &h_features_tensor);
  for (uint64_t i = 0; i < correct_out_features.size(); i++) {
    float tmp = std::fabs(static_cast<float>(correct_out_features[i] -
                                             h_features_tensor.data<T>()[i]));
    ASSERT_LT(tmp, diff);
  }
#endif
}

TEST(DEV_API, sparse_conv3d) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 4;
  std::vector<std::vector<int>> indices = {
      {0, 0, 0, 0}, {0, 2, 0, 2}, {3, 2, 2, 3}, {3, 2, 3, 2}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {-0.2883, 0.0287, 0.2864, -0.0992};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.4721, 0.2292, 0.9751, 0.8616, 0.5784, 0.9178, 0.8727, 0.1659, 0.4455,

      0.0189, 0.4646, 0.4472, 0.1991, 0.8968, 0.3717, 0.0051, 0.6963, 0.2690,

      0.7473, 0.5403, 0.5391, 0.0796, 0.4734, 0.9097, 0.1712, 0.6237, 0.8837};

  std::vector<std::vector<int>> out_indices = {{0, 0, 0, 0, 0, 0, 0, 0},
                                               {0, 0, 0, 0, 1, 1, 1, 1},
                                               {0, 0, 1, 1, 0, 0, 1, 1},
                                               {0, 1, 0, 1, 0, 1, 0, 1}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {
      0.0254, 0.1455, -0.0615, 0.0862, 0.0077, 0.0200, -0.0160, -0.0433};

  TestConv3d<float>(indices_flatten,
                    features,
                    x_dims,
                    kernel,
                    kernel_dims,
                    out_indices_flatten,
                    out_features,
                    out_dims,
                    non_zero_num,
                    paddings,
                    strides,
                    dilations);

  // test fp16
  using phi::dtype::float16;
  std::vector<float16> features_fp16(features.size()),
      out_features_fp16(out_features.size()), kernel_fp16(kernel.size());
  cast<float, float16>(features, &features_fp16);
  cast<float, float16>(out_features, &out_features_fp16);
  cast<float, float16>(kernel, &kernel_fp16);

  TestConv3d<float16>(indices_flatten,
                      features_fp16,
                      x_dims,
                      kernel_fp16,
                      kernel_dims,
                      out_indices_flatten,
                      out_features_fp16,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
}

TEST(DEV_API, sparse_conv3d_batch) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {2, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {2, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 8;
  std::vector<std::vector<int>> indices = {{0, 0, 0, 0, 1, 1, 1, 1},
                                           {0, 2, 0, 2, 0, 2, 0, 2},
                                           {3, 2, 2, 3, 3, 2, 2, 3},
                                           {3, 2, 3, 2, 3, 2, 3, 2}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {
      -0.2883, 0.0287, 0.2864, -0.0992, -0.2883, 0.0287, 0.2864, -0.0992};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.4721, 0.2292, 0.9751, 0.8616, 0.5784, 0.9178, 0.8727, 0.1659, 0.4455,

      0.0189, 0.4646, 0.4472, 0.1991, 0.8968, 0.3717, 0.0051, 0.6963, 0.2690,

      0.7473, 0.5403, 0.5391, 0.0796, 0.4734, 0.9097, 0.1712, 0.6237, 0.8837};

  std::vector<std::vector<int>> out_indices = {
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
      {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
      {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},
      {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {0.0254,
                                     0.1455,
                                     -0.0615,
                                     0.0862,
                                     0.0077,
                                     0.0200,
                                     -0.0160,
                                     -0.0433,
                                     0.0254,
                                     0.1455,
                                     -0.0615,
                                     0.0862,
                                     0.0077,
                                     0.0200,
                                     -0.0160,
                                     -0.0433};

  TestConv3d<float>(indices_flatten,
                    features,
                    x_dims,
                    kernel,
                    kernel_dims,
                    out_indices_flatten,
                    out_features,
                    out_dims,
                    non_zero_num,
                    paddings,
                    strides,
                    dilations);

  using phi::dtype::float16;
  std::vector<float16> features_fp16(features.size()),
      out_features_fp16(out_features.size()), kernel_fp16(kernel.size());
  cast<float, float16>(features, &features_fp16);
  cast<float, float16>(out_features, &out_features_fp16);
  cast<float, float16>(kernel, &kernel_fp16);

  TestConv3d<float16>(indices_flatten,
                      features_fp16,
                      x_dims,
                      kernel_fp16,
                      kernel_dims,
                      out_indices_flatten,
                      out_features_fp16,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
}

TEST(DEV_API, sparse_conv3d_stride) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 1, 1, 1, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {2, 2, 2};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<std::vector<int>> indices = {
      {0, 0, 0}, {0, 2, 0}, {3, 2, 2}, {3, 2, 3}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {-0.28833008, 0.02873230, 0.28637695};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.45043945, 0.47216797, 0.22924805, 0.97509766, 0.86181641, 0.57861328,
      0.91796875, 0.87255859, 0.16589355, 0.44555664, 0.01889038, 0.46459961,
      0.44726562, 0.19909668, 0.89697266, 0.37158203, 0.00513077, 0.69628906,
      0.26904297, 0.74707031, 0.54003906, 0.5390625,  0.07958984, 0.47338867,
      0.90966797, 0.17126465, 0.62353516};

  std::vector<std::vector<int>> out_indices = {{0, 0, 0, 0}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {0.01791};

  TestConv3d<float>(indices_flatten,
                    features,
                    x_dims,
                    kernel,
                    kernel_dims,
                    out_indices_flatten,
                    out_features,
                    out_dims,
                    non_zero_num,
                    paddings,
                    strides,
                    dilations);

  // test fp16
  using phi::dtype::float16;
  std::vector<float16> features_fp16(features.size()),
      out_features_fp16(out_features.size()), kernel_fp16(kernel.size());
  cast<float, float16>(features, &features_fp16);
  cast<float, float16>(out_features, &out_features_fp16);
  cast<float, float16>(kernel, &kernel_fp16);

  TestConv3d<float16>(indices_flatten,
                      features_fp16,
                      x_dims,
                      kernel_fp16,
                      kernel_dims,
                      out_indices_flatten,
                      out_features_fp16,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
}

TEST(DEV_API, sparse_conv3d_dilation) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 6, 6, 6, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {2, 2, 2};

  const int non_zero_num = 3;
  std::vector<std::vector<int>> indices = {
      {0, 0, 0}, {2, 3, 3}, {2, 3, 3}, {5, 2, 0}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {-0.78710938, -0.64746094, 0.98828125};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.20617676, 0.99365234, 0.16760254, 0.30639648, 0.41479492, 0.75732422,
      0.65625,    0.48535156, 0.72167969, 0.56005859, 0.5,        0.3581543,
      0.20324707, 0.88769531, 0.81298828, 0.58398438, 0.30810547, 0.12634277,
      0.70507812, 0.38720703, 0.34814453, 0.02690125, 0.80273438, 0.90625,
      0.2277832,  0.4362793,  0.44482422};

  std::vector<std::vector<int>> out_indices = {{0, 0, 0, 1, 0, 1, 1, 0}};
  std::vector<int> out_indices_flatten = flatten(out_indices);

  std::vector<float> out_features = {-0.64014, -0.37402};

  TestConv3d<float>(indices_flatten,
                    features,
                    x_dims,
                    kernel,
                    kernel_dims,
                    out_indices_flatten,
                    out_features,
                    out_dims,
                    non_zero_num,
                    paddings,
                    strides,
                    dilations);

  // test fp16
  using phi::dtype::float16;
  std::vector<float16> features_fp16(features.size()),
      out_features_fp16(out_features.size()), kernel_fp16(kernel.size());
  cast<float, float16>(features, &features_fp16);
  cast<float, float16>(out_features, &out_features_fp16);
  cast<float, float16>(kernel, &kernel_fp16);

  TestConv3d<float16>(indices_flatten,
                      features_fp16,
                      x_dims,
                      kernel_fp16,
                      kernel_dims,
                      out_indices_flatten,
                      out_features_fp16,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
}

TEST(DEV_API, sparse_conv3d_padding) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 3, 3, 3, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 3, 3, 3, out_channels};
  std::vector<int> paddings = {1, 1, 1};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 1;
  std::vector<std::vector<int>> indices = {{0, 1, 0, 0}};
  std::vector<int> indices_flatten = flatten(indices);

  std::vector<float> features = {-0.79394531};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.34375,    0.22485352, 0.65820312, 0.75048828, 0.21411133, 0.17370605,
      0.85546875, 0.53076172, 0.28833008, 0.71044922, 0.00659943, 0.45922852,
      0.19372559, 0.64599609, 0.78808594, 0.49316406, 0.62646484, 0.40649414,
      0.62744141, 0.5703125,  0.23144531, 0.50048828, 0.31835938, 0.90869141,
      0.38208008, 0.60449219, 0.09075928};

  std::vector<int> out_indices_flatten = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
      0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  std::vector<float> out_features = {-0.25269,
                                     -0.39746,
                                     -0.45288,
                                     -0.49805,
                                     -0.5127,
                                     -0.15381,
                                     -0.00524,
                                     -0.56396,
                                     -0.17004,
                                     -0.5957,
                                     -0.17847,
                                     -0.27295};

  TestConv3d<float>(indices_flatten,
                    features,
                    x_dims,
                    kernel,
                    kernel_dims,
                    out_indices_flatten,
                    out_features,
                    out_dims,
                    non_zero_num,
                    paddings,
                    strides,
                    dilations);

  // test fp16
  using phi::dtype::float16;
  std::vector<float16> features_fp16(features.size()),
      out_features_fp16(out_features.size()), kernel_fp16(kernel.size());
  cast<float, float16>(features, &features_fp16);
  cast<float, float16>(out_features, &out_features_fp16);
  cast<float, float16>(kernel, &kernel_fp16);

  TestConv3d<float16>(indices_flatten,
                      features_fp16,
                      x_dims,
                      kernel_fp16,
                      kernel_dims,
                      out_indices_flatten,
                      out_features_fp16,
                      out_dims,
                      non_zero_num,
                      paddings,
                      strides,
                      dilations);
}

}  // namespace tests
}  // namespace phi
