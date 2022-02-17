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

#include <set>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/sparse/convolution_kernel.h"

namespace pten {
namespace sparse {

inline bool Check(const int& x, const int& y, const int& z, const DDim& dims) {
  if (x >= 0 && x < dims[3] && y >= 0 && y < dims[2] && z >= 0 && z < dims[1]) {
    return true;
  }
  return false;
}

inline int PointToIndex(const int& batch,
                        const int& x,
                        const int& y,
                        const int& z,
                        const DDim& dims) {
  return batch * dims[1] * dims[2] * dims[3] + z * dims[2] * dims[3] +
         y * dims[3] + x;
}

inline void IndexToPoint(
    const int index, const DDim& dims, int* batch, int* x, int* y, int* z) {
  int n = index;
  *x = n % dims[3];
  n /= dims[3];
  *y = n % dims[2];
  n /= dims[2];
  *z = n % dims[1];
  n /= dims[1];
  *batch = n;
}

// such as: kernel(3, 3, 3), kernel_size = 27
// counter_per_weight: (kernel_size)
// TODO(zhangkaihuo): optimize performance with multithreading
template <typename T, typename Context>
void ProductRuleBook(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const DDim& out_dims,
                     DenseTensor* rulebook,
                     DenseTensor* counter_per_kernel) {
  const auto place = dev_ctx.GetPlace();
  const auto& kernel_dims = kernel.dims();
  const int64_t non_zero_num = x.nnz();
  const auto& non_zero_indices = x.non_zero_indices();
  const int* indices_ptr = non_zero_indices.data<int>();
  int* counter_ptr = counter_per_kernel->mutable_data<int>(place);
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  memset(counter_ptr, 0, kernel_size * sizeof(int));

  int kernel_index = 0, rulebook_len = 0;
  for (int kernel_z = 0; kernel_z < kernel_dims[0]; kernel_z++) {
    for (int kernel_y = 0; kernel_y < kernel_dims[1]; kernel_y++) {
      for (int kernel_x = 0; kernel_x < kernel_dims[2]; kernel_x++) {
        for (int64_t i = 0; i < non_zero_num; i++) {
          // indices_ptr[i] is batch
          int in_z = indices_ptr[i + non_zero_num];
          int in_y = indices_ptr[i + 2 * non_zero_num];
          int in_x = indices_ptr[i + 3 * non_zero_num];
          int out_z =
              (in_z + paddings[0] - kernel_z * dilations[0]) / strides[0];
          int out_y =
              (in_y + paddings[1] - kernel_y * dilations[1]) / strides[1];
          int out_x =
              (in_x + paddings[2] - kernel_x * dilations[2]) / strides[2];
          if (Check(out_x, out_y, out_z, out_dims)) {
            counter_ptr[kernel_index] += 1;
            ++rulebook_len;
          }
        }
        ++kernel_index;
      }
    }
  }

  rulebook->ResizeAndAllocate({3, rulebook_len});
  int* rulebook_ptr = rulebook->mutable_data<int>(place);
  int rulebook_index = 0;
  kernel_index = 0;

  for (int kernel_z = 0; kernel_z < kernel_dims[0]; kernel_z++) {
    for (int kernel_y = 0; kernel_y < kernel_dims[1]; kernel_y++) {
      for (int kernel_x = 0; kernel_x < kernel_dims[2]; kernel_x++) {
        for (int64_t i = 0; i < non_zero_num; i++) {
          int batch = indices_ptr[i];
          int in_z = indices_ptr[i + non_zero_num];
          int in_y = indices_ptr[i + 2 * non_zero_num];
          int in_x = indices_ptr[i + 3 * non_zero_num];
          int out_z =
              (in_z + paddings[0] - kernel_z * dilations[0]) / strides[0];
          int out_y =
              (in_y + paddings[1] - kernel_y * dilations[1]) / strides[1];
          int out_x =
              (in_x + paddings[2] - kernel_x * dilations[2]) / strides[2];
          if (Check(out_x, out_y, out_z, out_dims)) {
            rulebook_ptr[rulebook_index] = kernel_index;
            rulebook_ptr[rulebook_index + rulebook_len] = i;  // in_i
            rulebook_ptr[rulebook_index + rulebook_len * 2] = PointToIndex(
                batch, out_x, out_y, out_z, out_dims);  // out_index
            ++rulebook_index;
          }
        }
        ++kernel_index;
      }
    }
  }
}

template <typename T, typename Context>
void UpdateRulebookAndOutIndex(const Context& dev_ctx,
                               const SparseCooTensor& x,
                               const int kernel_size,
                               const DDim& out_dims,
                               DenseTensor* rulebook,
                               SparseCooTensor* out) {
  std::set<int> out_indexs;
  int n = rulebook->dims()[1];
  int* rulebook_ptr = rulebook->data<int>();
  for (int i = 0; i < n; i++) {
    out_indexs.insert(rulebook_ptr[i + n * 2]);
  }

  int out_non_zero_num = out_indexs.size();
  const int64_t sparse_dim = 4;
  DenseTensorMeta indices_meta(
      DataType::INT32, {sparse_dim, out_non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, x.dims()[4]}, x.layout());
  pten::DenseTensor out_indices = pten::Empty(dev_ctx, std::move(indices_meta));
  pten::DenseTensor out_values = pten::Empty(dev_ctx, std::move(values_meta));
  int* out_indices_ptr = out_indices.mutable_data<int>(dev_ctx.GetPlace());
  int i = 0;
  for (auto it = out_indexs.begin(); it != out_indexs.end(); it++, i++) {
    const int index = *it;
    int batch, x, y, z;
    IndexToPoint(index, out_dims, &batch, &x, &y, &z);
    out_indices_ptr[i] = batch;
    out_indices_ptr[i + out_non_zero_num] = z;
    out_indices_ptr[i + out_non_zero_num * 2] = y;
    out_indices_ptr[i + out_non_zero_num * 3] = x;
  }
  for (i = 0; i < n; i++) {
    int out_index = rulebook_ptr[i + n * 2];
    rulebook_ptr[i + n * 2] =
        std::distance(out_indexs.begin(), out_indexs.find(out_index));
  }

  out->SetMember(out_indices, out_values, out_dims, true);
}

template <typename T>
void Gather(
    const T* x, const int* indexs, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    int real_i = indexs[i];
    memcpy(out + i * channels, x + real_i * channels, channels * sizeof(T));
  }
}

template <typename T>
void Scatter(
    const T* x, const int* indexs, const int n, const int channels, T* out) {
  for (int i = 0; i < n; i++) {
    int real_i = indexs[i];
    for (int j = 0; j < channels; j++) {
      out[real_i * channels + j] += x[i * channels + j];
    }
  }
}

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
**/
template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const SparseCooTensor& x,
                  const DenseTensor& kernel,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const int groups,
                  SparseCooTensor* out) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& place = dev_ctx.GetPlace();
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  GetOutShape(x_dims, kernel_dims, paddings, dilations, strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensor rulebook = pten::Empty<int, Context>(dev_ctx);
  DenseTensor counter_per_kernel =
      pten::Empty(dev_ctx, std::move(counter_meta));

  ProductRuleBook<T, Context>(dev_ctx,
                              x,
                              kernel,
                              paddings,
                              dilations,
                              strides,
                              out_dims,
                              &rulebook,
                              &counter_per_kernel);

  UpdateRulebookAndOutIndex<T>(
      dev_ctx, x, kernel_size, out_dims, &rulebook, out);

  int n = rulebook.dims()[1];
  const int* counter_ptr = counter_per_kernel.data<int>();

  // 2. gather
  DenseTensorMeta in_features_meta(
      x.dtype(), {n, in_channels}, DataLayout::NCHW);
  DenseTensorMeta out_features_meta(
      x.dtype(), {n, out_channels}, DataLayout::NCHW);
  pten::DenseTensor in_features =
      pten::Empty(dev_ctx, std::move(in_features_meta));
  pten::DenseTensor out_features =
      pten::Empty(dev_ctx, std::move(out_features_meta));
  T* in_features_ptr = in_features.mutable_data<T>(place);
  T* out_features_ptr = out_features.mutable_data<T>(place);

  Gather<T>(x.non_zero_elements().data<T>(),
            rulebook.data<int>() + n,
            n,
            in_channels,
            in_features_ptr);

  // 3. call gemm for every werght
  auto blas = paddle::operators::math::GetBlas<Context, T>(dev_ctx);
  std::vector<int> offsets(kernel_size + 1);
  int offset = 0;
  for (int i = 0; i < kernel_size; i++) {
    offsets[i] = offset;
    offset += counter_ptr[i];
  }
  offsets[kernel_size] = offset;

  const T* kernel_ptr = kernel.data<T>();
  for (int i = 0; i < kernel_size; i++) {
    if (counter_ptr[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = counter_ptr[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
    T* tmp_in_ptr = in_features_ptr + offsets[i] * in_channels;
    const T* tmp_kernel_ptr = kernel_ptr + i * K * N;
    T* tmp_out_ptr = out_features_ptr + offsets[i] * out_channels;
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              tmp_in_ptr,
              tmp_kernel_ptr,
              static_cast<T>(0),
              tmp_out_ptr);
  }

  // 4. scatter
  T* out_values_ptr = out->mutable_non_zero_elements()->mutable_data<T>(place);
  memset(out_values_ptr, 0, sizeof(T) * out->nnz() * out_channels);
  Scatter<T>(out_features_ptr,
             rulebook.data<int>() + n * 2,
             n,
             out_channels,
             out_values_ptr);
}

}  // namespace sparse
}  // namespace pten

PT_REGISTER_KERNEL(conv3d,
                   CPU,
                   ALL_LAYOUT,
                   pten::sparse::Conv3dKernel,
                   float,
                   double,
                   pten::dtype::float16) {}
