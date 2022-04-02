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
#include "paddle/phi/kernels/sparse/sparse_pool_grad_kernel.h"
#include "paddle/phi/kernels/sparse/sparse_pool_kernel.h"

#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace tests {

template <typename T1, typename T2>
std::vector<T2> cast(const std::vector<T1>& in) {
  std::vector<T2> out(in.size());
  for (uint64_t i = 0; i < in.size(); i++) {
    out[i] = static_cast<T2>(in[i]);
  }
  return out;
}
template <typename T, typename IntT = int>
void TestMaxPoolBase(const std::vector<IntT>& indices,
                     const std::vector<T>& features,
                     const DDim& x_dims,
                     const std::vector<IntT>& correct_out_indices,
                     const std::vector<T>& correct_out_features,
                     const DDim& correct_out_dims,
                     const int non_zero_num,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& strides,
                     const std::vector<int>& dilations,
                     const float diff = 1e-3,
                     const bool backward = false,
                     const std::vector<T> features_grad = {}) {
  phi::CPUContext dev_ctx_cpu;
  dev_ctx_cpu.SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  dev_ctx_cpu.SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  dev_ctx_cpu.Init();

  const int in_channels = x_dims[4];
  const int out_channels = in_channels;

  auto indices_dtype = paddle::experimental::CppTypeToDataType<IntT>::Type();
  DenseTensor indices_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(indices_dtype, {4, non_zero_num}, DataLayout::NCHW));
  memcpy(indices_tensor.data<IntT>(),
         indices.data(),
         indices.size() * sizeof(IntT));
  DenseTensor features_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(paddle::experimental::CppTypeToDataType<T>::Type(),
                      {non_zero_num, in_channels},
                      DataLayout::NHWC));
  memcpy(
      features_tensor.data<T>(), features.data(), features.size() * sizeof(T));

  SparseCooTensor x_tensor(indices_tensor, features_tensor, x_dims);

  auto f_verify = [&](const T* real_data, const std::vector<T>& correct_data) {
    for (uint64_t i = 0; i < correct_data.size(); i++) {
      float tmp = std::fabs(static_cast<float>(correct_data[i] - real_data[i]));
      ASSERT_LT(tmp, diff);
    }
  };

  if (!std::is_same<T, phi::dtype::float16>::value) {
    DenseTensor rulebook;
    SparseCooTensor out = sparse::MaxPool<T>(dev_ctx_cpu,
                                             x_tensor,
                                             kernel_sizes,
                                             paddings,
                                             dilations,
                                             strides,
                                             &rulebook);

    ASSERT_EQ(correct_out_dims.size(), out.dims().size());
    for (int i = 0; i < correct_out_dims.size(); i++) {
      ASSERT_EQ(correct_out_dims[i], out.dims()[i]);
    }
    ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, out.nnz());

    int cmp_indices = memcmp(correct_out_indices.data(),
                             out.non_zero_indices().data<IntT>(),
                             correct_out_indices.size() * sizeof(IntT));
    ASSERT_EQ(cmp_indices, 0);

    f_verify(out.non_zero_elements().data<T>(), correct_out_features);

    if (backward) {
      SparseCooTensor x_grad = sparse::MaxPoolGrad<T>(
          dev_ctx_cpu, x_tensor, rulebook, out, out, kernel_sizes);
      f_verify(x_grad.non_zero_elements().data<T>(), features_grad);
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
  dev_ctx_gpu.SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
  dev_ctx_gpu.PartialInitWithAllocator();

  DenseTensor d_indices_tensor = phi::Empty(
      dev_ctx_gpu,
      DenseTensorMeta(indices_dtype, {4, non_zero_num}, DataLayout::NCHW));
  phi::Copy(
      dev_ctx_gpu, indices_tensor, phi::GPUPlace(), true, &d_indices_tensor);

  DenseTensor d_features_tensor =
      phi::EmptyLike<T>(dev_ctx_gpu, features_tensor);
  phi::Copy(
      dev_ctx_gpu, features_tensor, phi::GPUPlace(), true, &d_features_tensor);

  SparseCooTensor d_x_tensor(d_indices_tensor, d_features_tensor, x_dims);

  DenseTensor d_rulebook;
  SparseCooTensor d_out = sparse::MaxPool<T>(dev_ctx_gpu,
                                             d_x_tensor,
                                             kernel_sizes,
                                             paddings,
                                             dilations,
                                             strides,
                                             &d_rulebook);

  ASSERT_EQ(correct_out_dims.size(), d_out.dims().size());
  ASSERT_EQ((int64_t)correct_out_features.size() / out_channels, d_out.nnz());
  for (int i = 0; i < correct_out_dims.size(); i++) {
    ASSERT_EQ(correct_out_dims[i], d_out.dims()[i]);
  }

  DenseTensor h_indices_tensor = phi::Empty(
      dev_ctx_cpu,
      DenseTensorMeta(indices_dtype, {4, d_out.nnz()}, DataLayout::NCHW));
  phi::Copy(dev_ctx_gpu,
            d_out.non_zero_indices(),
            phi::CPUPlace(),
            true,
            &h_indices_tensor);

  int cmp_indices2 = memcmp(correct_out_indices.data(),
                            h_indices_tensor.data<IntT>(),
                            correct_out_indices.size() * sizeof(IntT));
  ASSERT_EQ(cmp_indices2, 0);

  DenseTensor h_features_tensor =
      phi::EmptyLike<T>(dev_ctx_cpu, d_out.non_zero_elements());

  phi::Copy(dev_ctx_gpu,
            d_out.non_zero_elements(),
            phi::CPUPlace(),
            true,
            &h_features_tensor);
  f_verify(h_features_tensor.data<T>(), correct_out_features);

  if (backward) {
    SparseCooTensor x_grad = sparse::MaxPoolGrad<T>(
        dev_ctx_gpu, d_x_tensor, d_rulebook, d_out, d_out, kernel_sizes);
    DenseTensor h_features_grad =
        phi::EmptyLike<T>(dev_ctx_cpu, x_grad.non_zero_elements());
    phi::Copy(dev_ctx_gpu,
              x_grad.non_zero_elements(),
              phi::CPUPlace(),
              true,
              &h_features_grad);
    f_verify(h_features_grad.data<T>(), features_grad);
  }
#endif
}

template <typename IntT = int>
void TestMaxPool(const std::vector<IntT>& indices,
                 const std::vector<float>& features,
                 const DDim& x_dims,
                 const std::vector<IntT>& correct_out_indices,
                 const std::vector<float>& correct_out_features,
                 const DDim& correct_out_dims,
                 const int non_zero_num,
                 const std::vector<int>& kernel_sizes,
                 const std::vector<int>& paddings,
                 const std::vector<int>& strides,
                 const std::vector<int>& dilations,
                 const float diff = 1e-3,
                 const bool backward = false,
                 const std::vector<float> features_grad = {}) {
  // test float
  TestMaxPoolBase<float, IntT>(indices,
                               features,
                               x_dims,
                               correct_out_indices,
                               correct_out_features,
                               correct_out_dims,
                               non_zero_num,
                               kernel_sizes,
                               paddings,
                               strides,
                               dilations,
                               diff,
                               backward,
                               features_grad);
  // test double
  TestMaxPoolBase<double, IntT>(indices,
                                cast<float, double>(features),
                                x_dims,
                                correct_out_indices,
                                cast<float, double>(correct_out_features),
                                correct_out_dims,
                                non_zero_num,
                                kernel_sizes,
                                paddings,
                                strides,
                                dilations,
                                diff,
                                backward,
                                cast<float, double>(features_grad));
}

TEST(DEV_API, sparse_maxpool) {
  const int channels = 1;
  DDim x_dims = {1, 1, 4, 4, channels};
  DDim out_dims = {1, 1, 2, 2, channels};
  std::vector<int> kernel_sizes = {1, 3, 3};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices = {0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 2};
  std::vector<float> features = {1, 2, 3};
  std::vector<int> out_indices = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
  };
  std::vector<float> out_features = {2, 2, 3, 3};
  std::vector<float> x_grad = {0, 4, 6};

  TestMaxPool(indices,
              features,
              x_dims,
              out_indices,
              out_features,
              out_dims,
              non_zero_num,
              kernel_sizes,
              paddings,
              strides,
              dilations,
              1e-6,
              true,
              x_grad);
}

TEST(DEV_API, sparse_maxpool_stride) {
  const int channels = 1;
  DDim x_dims = {1, 1, 4, 4, channels};
  DDim out_dims = {1, 1, 1, 1, channels};
  std::vector<int> kernel_sizes = {1, 3, 3};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {2, 2, 2};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices = {0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 2};
  std::vector<float> features = {1, 2, 3};
  std::vector<int> out_indices = {0, 0, 0, 0};
  std::vector<float> out_features = {2};
  std::vector<float> x_grad = {0, 2, 0};

  TestMaxPool(indices,
              features,
              x_dims,
              out_indices,
              out_features,
              out_dims,
              non_zero_num,
              kernel_sizes,
              paddings,
              strides,
              dilations,
              1e-6,
              true,
              x_grad);
}

TEST(DEV_API, sparse_maxpool_channel) {
  const int channels = 2;
  DDim x_dims = {1, 1, 4, 4, channels};
  DDim out_dims = {1, 1, 2, 2, channels};
  std::vector<int> kernel_sizes = {1, 3, 3};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices = {0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 2};
  std::vector<float> features = {1, 1, 2, 2, 3, 3};
  std::vector<int> out_indices = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
  };
  std::vector<float> out_features = {2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<float> x_grad = {0, 0, 4, 4, 6, 6};

  TestMaxPool(indices,
              features,
              x_dims,
              out_indices,
              out_features,
              out_dims,
              non_zero_num,
              kernel_sizes,
              paddings,
              strides,
              dilations,
              1e-6,
              true,
              x_grad);
}

TEST(DEV_API, sparse_maxpool3d) {
  const int channels = 2;
  DDim x_dims = {1, 5, 4, 4, channels};
  DDim out_dims = {1, 3, 2, 2, channels};
  std::vector<int> kernel_sizes = {3, 3, 3};
  std::vector<int> paddings = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  std::vector<int> dilations = {1, 1, 1};

  const int non_zero_num = 3;
  std::vector<int> indices = {0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 1, 2};
  std::vector<float> features = {1, 1, 2, 2, 3, 3};
  std::vector<int> out_indices = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
  };
  std::vector<float> out_features = {2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<float> x_grad = {0, 0, 4, 4, 6, 6};

  TestMaxPool(indices,
              features,
              x_dims,
              out_indices,
              out_features,
              out_dims,
              non_zero_num,
              kernel_sizes,
              paddings,
              strides,
              dilations,
              1e-6,
              true,
              x_grad);
}

}  // namespace tests
}  // namespace phi
