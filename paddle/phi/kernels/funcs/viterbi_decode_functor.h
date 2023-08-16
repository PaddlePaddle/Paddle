// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

static std::vector<DenseTensor> Unbind(const DenseTensor& in) {
  int64_t size = in.dims()[0];
  std::vector<DenseTensor> tensors(size);
  for (int64_t i = 0; i < size; ++i) {
    tensors[i] = in.Slice(i, i + 1);
  }
  return tensors;
}

template <typename T, typename Functor, typename OutT = T>
void SameDimsBinaryOP(const DenseTensor& lhs,
                      const DenseTensor& rhs,
                      DenseTensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  OutT* out_ptr = out->data<OutT>();
  Functor functor;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < out->numel(); ++i) {
    out_ptr[i] = functor(lhs_ptr[i], rhs_ptr[i]);
  }
}

template <bool is_multi_threads>
struct GetInputIndex {
  void operator()(const std::vector<int>& lhs_dims,
                  const std::vector<int>& rhs_dims,
                  const std::vector<int>& output_dims UNUSED,
                  const std::vector<int>& lhs_strides,
                  const std::vector<int>& rhs_strides,
                  const std::vector<int>& output_strides,
                  int output_idx,
                  int* index_array UNUSED,
                  int* lhs_idx,
                  int* rhs_idx) {
    int out_dims_size = output_strides.size();
    for (int j = 0; j < out_dims_size; ++j) {
      int curr_idx = output_idx / output_strides[j];
      output_idx %= output_strides[j];
      *lhs_idx += (lhs_dims[j] > 1) ? curr_idx * lhs_strides[j] : 0;
      *rhs_idx += (rhs_dims[j] > 1) ? curr_idx * rhs_strides[j] : 0;
    }
  }
};

template <typename T, typename Functor, bool is_multi_threads = false>
void SimpleBroadcastBinaryOP(const DenseTensor& lhs,
                             const DenseTensor& rhs,
                             DenseTensor* out) {
  const T* lhs_ptr = lhs.data<T>();
  const T* rhs_ptr = rhs.data<T>();
  T* out_ptr = out->data<T>();
  int out_size = static_cast<int>(out->dims().size());
  std::vector<int> out_dims(out_size);
  std::vector<int> lhs_dims(out_size);
  std::vector<int> rhs_dims(out_size);
  std::copy(lhs.dims().Get(), lhs.dims().Get() + out_size, lhs_dims.data());
  std::copy(rhs.dims().Get(), rhs.dims().Get() + out_size, rhs_dims.data());
  std::copy(out->dims().Get(), out->dims().Get() + out_size, out_dims.data());
  std::vector<int> output_strides(out_size, 1);
  std::vector<int> lhs_strides(out_size, 1);
  std::vector<int> rhs_strides(out_size, 1);
  std::vector<int> index_array(out_size, 0);
  // calculate strides
  for (int i = out_size - 2; i >= 0; --i) {
    output_strides[i] = output_strides[i + 1] * out_dims[i + 1];
    lhs_strides[i] = lhs_strides[i + 1] * lhs_dims[i + 1];
    rhs_strides[i] = rhs_strides[i + 1] * rhs_dims[i + 1];
  }
  Functor functor;
  GetInputIndex<is_multi_threads> get_input_index;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int i = 0; i < out->numel(); ++i) {
    int lhs_idx = 0;
    int rhs_idx = 0;
    get_input_index(lhs_dims,
                    rhs_dims,
                    out_dims,
                    lhs_strides,
                    rhs_strides,
                    output_strides,
                    i,
                    index_array.data(),
                    &lhs_idx,
                    &rhs_idx);
    out_ptr[i] = functor(lhs_ptr[lhs_idx], rhs_ptr[rhs_idx]);
  }
}

class TensorBuffer {
 public:
  explicit TensorBuffer(const DenseTensor& in) : buffer_(in), offset_(0) {
    buffer_.Resize({buffer_.numel()});
  }
  DenseTensor GetBufferBlock(std::initializer_list<int64_t> shape) {
    int64_t size = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    DenseTensor block = buffer_.Slice(offset_, offset_ + size);
    offset_ += size;
    block.Resize(shape);
    return block;
  }

 private:
  DenseTensor buffer_;  // need to resize 1-D Tensor
  int offset_;
};

}  // namespace funcs
}  // namespace phi
