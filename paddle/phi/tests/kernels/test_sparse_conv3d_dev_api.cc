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

#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/sparse/convolution_grad_kernel.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace tests {

std::vector<int> flatten(const std::vector<std::vector<int>>& in) {
  std::vector<int> out;
  if (in.size() == 0) return out;
  const int cols = in[0].size();
  out.resize(in.size() * cols);
  for (uint64_t i = 0; i < in.size(); i++) {
    memcpy(&out[i * cols], in[i].data(), cols * sizeof(int));
  }
  return out;
}

template <typename T1, typename T2>
std::vector<T2> cast(const std::vector<T1>& in) {
  std::vector<T2> out(in.size());
  for (uint64_t i = 0; i < in.size(); i++) {
    out[i] = static_cast<T2>(in[i]);
  }
  return out;
}

template <typename T>
void TestConv3dBase(const std::vector<int>& indices,
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
                    const std::vector<int>& dilations,
                    const float diff = 1e-3,
                    const bool backward = false,
                    const std::vector<T> features_grad = {},
                    const std::vector<T> kernel_grad = {}) {
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  DenseTensor indices_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(DataType::INT32, {4, non_zero_num}, DataLayout::NCHW));
  dev_ctx_cpu.Alloc(&indices_tensor,
                    indices_tensor.dtype(),
                    sizeof(int) * indices_tensor.numel());
  memcpy(
      indices_tensor.data<int>(), indices.data(), indices.size() * sizeof(int));
  DenseTensor features_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NHWC));
  dev_ctx_cpu.Alloc(&features_tensor,
                    features_tensor.dtype(),
                    features_tensor.numel() * sizeof(T));
  memcpy(
      features_tensor.data<T>(), features.data(), features.size() * sizeof(T));

  SparseCooTensor x_tensor(indices_tensor, features_tensor, x_dims);

  DenseTensor kernel_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      kernel_dims,
                      DataLayout::NHWC));
  dev_ctx_cpu.Alloc(
      &kernel_tensor, kernel_tensor.dtype(), kernel_tensor.numel() * sizeof(T));
  memcpy(kernel_tensor.data<T>(), kernel.data(), kernel.size() * sizeof(T));

  if (!std::is_same<T, phi::dtype::float16>::value) {
    DenseTensor rulebook = phi::Empty<int, phi::CPUContext>(dev_ctx_cpu);
    SparseCooTensor out = sparse::Conv3d<T>(dev_ctx_cpu,
                                            x_tensor,
                                            kernel_tensor,
                                            paddings,
                                            dilations,
                                            strides,
                                            1,
                                            &rulebook);

    ASSERT_EQ(correct_out_dims.size(), out.dims().size());
    for (int i = 0; i < correct_out_dims.size(); i++) {
      ASSERT_EQ(correct_out_dims[i], out.dims()[i]);
    }
    ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, out.nnz());

    int cmp_indices = memcmp(correct_out_indices.data(),
                             out.non_zero_indices().data<int>(),
                             correct_out_indices.size() * sizeof(int));
    ASSERT_EQ(cmp_indices, 0);

    auto f_verify = [&](const T* real_data,
                        const std::vector<T>& correct_data) {
      for (uint64_t i = 0; i < correct_data.size(); i++) {
        float tmp =
            std::fabs(static_cast<float>(correct_data[i] - real_data[i]));
        ASSERT_LT(tmp, diff);
      }
    };

    f_verify(out.non_zero_elements().data<T>(), correct_out_features);

    if (backward) {
      std::vector<DenseTensor> grads = sparse::Conv3dGrad<T>(dev_ctx_cpu,
                                                             x_tensor,
                                                             rulebook,
                                                             kernel_tensor,
                                                             out,
                                                             paddings,
                                                             dilations,
                                                             strides,
                                                             1);
      f_verify(grads[0].data<T>(), features_grad);
      f_verify(grads[1].data<T>(), kernel_grad);
    }
  }
}

void TestConv3d(const std::vector<int>& indices,
                const std::vector<float>& features,
                const DDim& x_dims,
                const std::vector<float>& kernel,
                const DDim& kernel_dims,
                const std::vector<int>& correct_out_indices,
                const std::vector<float>& correct_out_features,
                const DDim& correct_out_dims,
                const int non_zero_num,
                const std::vector<int>& paddings,
                const std::vector<int>& strides,
                const std::vector<int>& dilations,
                const float diff = 1e-3,
                const bool backward = false,
                const std::vector<float> features_grad = {},
                const std::vector<float> kernel_grad = {}) {
  // test float
  TestConv3dBase<float>(indices,
                        features,
                        x_dims,
                        kernel,
                        kernel_dims,
                        correct_out_indices,
                        correct_out_features,
                        correct_out_dims,
                        non_zero_num,
                        paddings,
                        strides,
                        dilations,
                        diff,
                        backward,
                        features_grad,
                        kernel_grad);
  // test double
  TestConv3dBase<double>(indices,
                         cast<float, double>(features),
                         x_dims,
                         cast<float, double>(kernel),
                         kernel_dims,
                         correct_out_indices,
                         cast<float, double>(correct_out_features),
                         correct_out_dims,
                         non_zero_num,
                         paddings,
                         strides,
                         dilations,
                         diff,
                         backward,
                         cast<float, double>(features_grad),
                         cast<float, double>(kernel_grad));
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

  TestConv3d(indices_flatten,
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

  TestConv3d(indices_flatten,
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

  TestConv3d(indices_flatten,
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

  TestConv3d(indices_flatten,
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

  TestConv3d(indices_flatten,
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
}

TEST(DEV_API, sparse_conv2d) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 1, 5, 5, in_channels};
  DDim kernel_dims = {1, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 1, 3, 3, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices_flatten = {0, 0, 0, 0, 0, 0, 0, 4, 0, 3, 2, 4};

  std::vector<float> features = {-0.79394531, -0.3125, -0.55029297};
  // 3*3*3=27
  std::vector<float> kernel = {0.65820312,
                               0.75048828,
                               0.21411133,
                               0.17370605,
                               0.85546875,
                               0.53076172,
                               0.28833008,
                               0.71044922,
                               0.00659943};

  std::vector<int> out_indices_flatten = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 2, 2, 2, 1, 2, 0, 1, 2};

  std::vector<float> out_features = {
      -0.17004, -0.71338, -0.00206, -0.22205, -0.09009};

  TestConv3d(indices_flatten,
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
}

TEST(DEV_API, sparse_conv3d_backward) {
  const int in_channels = 1;
  const int out_channels = 1;
  DDim x_dims = {1, 4, 4, 4, in_channels};
  DDim kernel_dims = {3, 3, 3, in_channels, out_channels};
  DDim out_dims = {1, 2, 2, 2, out_channels};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 2;
  std::vector<int> indices_flatten = {0, 0, 0, 2, 3, 2, 3, 2};

  std::vector<float> features = {-0.28833008, 0.0287323};
  // 3*3*3=27
  std::vector<float> kernel = {
      0.64306641, 0.45043945, 0.47216797, 0.22924805, 0.97509766, 0.86181641,
      0.57861328, 0.91796875, 0.87255859, 0.16589355, 0.44555664, 0.01889038,
      0.46459961, 0.44726562, 0.19909668, 0.89697266, 0.37158203, 0.00513077,
      0.69628906, 0.26904297, 0.74707031, 0.54003906, 0.5390625,  0.07958984,
      0.47338867, 0.90966797, 0.17126465};

  std::vector<int> out_indices_flatten = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
                                          1, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  std::vector<float> out_features = {4.9200e-03,
                                     2.6140e-02,
                                     2.2900e-03,
                                     -2.3596e-01,
                                     1.5000e-04,
                                     1.0670e-02,
                                     5.7200e-03,
                                     1.2850e-02};

  std::vector<float> features_grad = {-0.20593, -0.09149};
  std::vector<float> kernel_grad = {
      0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,  0.000e+00,
      0.000e+00, 0.000e+00, 6.805e-02, 0.000e+00, 0.000e+00,  0.000e+00,
      0.000e+00, 3.700e-04, 1.600e-04, 0.000e+00, 3.100e-04,  0.000e+00,
      0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, -6.780e-03, 7.000e-05,
      0.000e+00, 7.500e-04, 1.400e-04};

  TestConv3d(indices_flatten,
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
             dilations,
             1e-3,
             true,
             features_grad,
             kernel_grad);
}

}  // namespace tests
}  // namespace phi
