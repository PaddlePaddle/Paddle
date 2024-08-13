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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

class DenseTensorUtils {
 public:
  static DenseTensorMeta* GetMutableMeta(DenseTensor* tensor) {
    return &(tensor->meta_);
  }

  static SparseTensorMeta* GetMutableMeta(SparseCooTensor* tensor) {
    return &(tensor->meta_);
  }

  static SparseTensorMeta* GetMutableMeta(SparseCsrTensor* tensor) {
    return &(tensor->meta_);
  }

  static const std::shared_ptr<phi::Allocation>& GetHolder(
      const DenseTensor& tensor) {
    return tensor.holder_;
  }

  static DenseTensor Slice(const DenseTensor& tensor,
                           int64_t begin_idx,
                           int64_t end_idx) {
    size_t bytes = tensor.numel() * SizeOf(tensor.dtype());
    PADDLE_ENFORCE_GE(tensor.capacity(),
                      bytes,
                      common::errors::InvalidArgument(
                          "The memory size %d should be enough to meet the "
                          "volume required by metadata %d.",
                          tensor.capacity(),
                          bytes));
    PADDLE_ENFORCE_GE(
        begin_idx,
        0,
        common::errors::OutOfRange("The start row index must be greater than 0."
                                   "But received the start index is d%.",
                                   begin_idx));
    PADDLE_ENFORCE_LE(
        end_idx,
        tensor.dims()[0],
        common::errors::OutOfRange("The end row index is out of bound."));
    PADDLE_ENFORCE_LT(
        begin_idx,
        end_idx,
        common::errors::InvalidArgument(
            "The start row index must be less than the end row index."
            "But received the start index = %d, the end index = %d.",
            begin_idx,
            end_idx));
    DenseTensor ret(tensor);
    if (tensor.dims()[0] != 1) {
      ret.meta_.dims[0] = end_idx - begin_idx;
      ret.meta_.offset = tensor.meta_.offset +
                         begin_idx * (tensor.numel() / tensor.dims()[0]) *
                             phi::SizeOf(tensor.dtype());
    }
    return ret;
  }
};

template <typename Context>
void Copy(const Context& dev_ctx,
          const DenseTensor& src,
          Place dst_place,
          bool blocking,
          DenseTensor* dst);

template <typename Context>
void Copy(const Context& dev_ctx,
          const SelectedRows& src,
          Place dst_place,
          bool blocking,
          SelectedRows* dst);

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCooTensor& src,
          Place dst_place,
          bool blocking,
          SparseCooTensor* dst);

template <typename Context>
void Copy(const Context& dev_ctx,
          const SparseCsrTensor& src,
          Place dst_place,
          bool blocking,
          SparseCsrTensor* dst);

template <typename Context>
void Copy(const Context& dev_ctx,
          const TensorArray& src,
          Place dst_place,
          bool blocking,
          TensorArray* dst);

template <typename T>
void TensorFromVector(const std::vector<T>& src,
                      const phi::DeviceContext& ctx,
                      phi::DenseTensor* dst);

template <typename T>
void TensorFromArray(const T* src,
                     const size_t& array_size,
                     const phi::DeviceContext& ctx,
                     phi::DenseTensor* dst);

template <typename T>
void TensorToVector(const phi::DenseTensor& src,
                    const phi::DeviceContext& ctx,
                    std::vector<T>* dst);

TEST_API phi::DenseTensor ReshapeToMatrix(const phi::DenseTensor& src,
                                          int num_col_dims);

template <typename T>
T GetValue(const phi::DenseTensor* x);

template <typename T, typename Context>
inline T GetValue(const Context& dev_ctx, const DenseTensor& x) {
  T value = static_cast<T>(0);
  if (x.place() != CPUPlace()) {
    DenseTensor cpu_x;
    Copy(dev_ctx, x, CPUPlace(), true, &cpu_x);
    value = cpu_x.data<T>()[0];
  } else {
    value = x.data<T>()[0];
  }
  return value;
}

template <typename T = int32_t>
std::vector<T> GetVectorFromTensor(const phi::DenseTensor* x);

}  // namespace phi
