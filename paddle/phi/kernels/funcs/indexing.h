// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/common/array.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/nonzero_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/split_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#endif

namespace phi {

namespace funcs {

static inline common::DDim InferSizeSymdimvector(const common::DDim& a,
                                                 const common::DDim& b) {
  auto dimsA = a.size();
  auto dimsB = b.size();
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  common::DDim expandedSizes = common::make_ddim(std::vector<int64_t>(ndim, 0));

  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    PADDLE_ENFORCE_EQ(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        true,
        common::errors::Fatal("The size of tensor a (",
                              sizeA,
                              ") must match the size of tensor b (",
                              sizeB,
                              ") at non-singleton dimension ",
                              i));

    expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
  }

  return expandedSizes;
}

template <typename T, typename Context>
std::vector<phi::DenseTensor*> ExpandTensors(
    const Context& dev_ctx, const std::vector<phi::DenseTensor*>& indices) {
  std::vector<phi::DenseTensor*> result;
  for (auto& index : indices) {
    if (index->dtype() == paddle::DataType::BOOL) {
      phi::DenseTensor bool_2_idx(phi::DataType::INT64);
      NonZeroKernel<bool, Context>(dev_ctx, *index, &bool_2_idx);
      for (int j = 0; j < index->dims().size(); j++) {
        SliceKernel<int64_t, Context>(
            dev_ctx, bool_2_idx, {1}, {j}, {j + 1}, {1}, {1}, index);
        result.emplace_back(index);
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

template <typename T, typename Context>
std::vector<phi::DenseTensor*> ExpandOutplace(
    const Context& dev_ctx, const std::vector<phi::DenseTensor*>& to_expand) {
  bool first = true;
  common::DDim sizes;
  for (size_t i = 0; i < to_expand.size(); i++) {
    if (!to_expand[i]->initialized()) {
      continue;
    } else if (first) {
      sizes = to_expand[i]->dims();
      first = false;
    } else {
      sizes = InferSizeSymdimvector(sizes, to_expand[i]->dims());
    }
  }

  std::vector<phi::DenseTensor*> result(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); i++) {
    if (!to_expand[i]->initialized()) {
      continue;
    } else if (to_expand[i]->dims() == sizes) {
      result[i] = to_expand[i];
    } else {
      if (to_expand[i]->dtype() == phi::DataType::INT32) {
        phi::DenseTensor tmp_idx(phi::DataType::INT64);
        ExpandKernel<int32_t, Context>(
            dev_ctx,
            *(to_expand[i]),
            IntArray(common::vectorize<int32_t>(sizes)),
            &tmp_idx);
        *(to_expand[i]) = tmp_idx;
        result[i] = to_expand[i];
      } else if (to_expand[i]->dtype() == phi::DataType::INT64) {
        phi::DenseTensor tmp_idx(phi::DataType::INT64);
        ExpandKernel<int64_t, Context>(
            dev_ctx,
            *(to_expand[i]),
            IntArray(common::vectorize<int64_t>(sizes)),
            &tmp_idx);
        *(to_expand[i]) = tmp_idx;
        result[i] = to_expand[i];
      } else {
        PADDLE_THROW(::common::errors::Unimplemented(
            "Index in Stride Mechanism must be int32_t, int64_t or bool"));
      }
    }
  }
  return result;
}

template <typename T, typename Context>
struct AdvancedIndex {
  AdvancedIndex(const Context& dev_ctx,
                const phi::DenseTensor& self,
                const std::vector<const phi::DenseTensor*>& orig);
  ~AdvancedIndex();
  phi::DenseTensor src;
  std::vector<phi::DenseTensor*> tmp_indices;
  std::vector<phi::DenseTensor*> indices;
  std::vector<int64_t> indexed_sizes;
  std::vector<int64_t> indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
  bool bool_case;
};

inline static phi::DenseTensor RestrideSrc(
    phi::DenseTensor* src,
    const int64_t& dims_before,
    const int64_t& dims_indexed,
    const std::vector<int64_t>& replacement_shape) {
  std::vector<int64_t> shape_vec = (common::vectorize<int64_t>(src->dims()));
  std::vector<int64_t> strides_vec =
      (common::vectorize<int64_t>(src->strides()));
  std::vector<int64_t>* shape = &shape_vec;
  std::vector<int64_t>* strides = &strides_vec;
  int64_t end = dims_before + dims_indexed;
  shape->erase(shape->begin() + dims_before, shape->begin() + end);
  strides->erase(strides->begin() + dims_before, strides->begin() + end);
  shape->insert(shape->begin() + dims_before,
                replacement_shape.begin(),
                replacement_shape.end());
  strides->insert(strides->begin() + dims_before, replacement_shape.size(), 0);
  auto meta = src->meta();
  meta.dims = common::make_ddim(*shape);
  meta.strides = common::make_ddim(*strides);
  meta.offset = src->offset();
  src->set_meta(meta);
  return *src;
}

inline static void ReshapeIndexer(phi::DenseTensor* index,
                                  const int64_t& dims_before,
                                  const int64_t& dims_after) {
  auto orig_shape = common::vectorize<int64_t>(index->dims());
  auto shape = std::vector<int64_t>{};
  shape.insert(shape.end(), dims_before, 1);
  shape.insert(shape.end(), orig_shape.begin(), orig_shape.end());
  shape.insert(shape.end(), dims_after, 1);
  index->Resize(common::make_ddim(shape));
}

template <typename T, typename Context>
inline AdvancedIndex<T, Context>::~AdvancedIndex() {
  for (phi::DenseTensor* ptr : tmp_indices) {
    delete ptr;
  }
}

template <typename T, typename Context>
inline AdvancedIndex<T, Context>::AdvancedIndex(
    const Context& dev_ctx,
    const phi::DenseTensor& self,
    const std::vector<const phi::DenseTensor*>& orig) {
  for (int i = 0; i < orig.size(); i++) {
    phi::DenseTensor* tmp = new phi::DenseTensor();
    *tmp = *(const_cast<phi::DenseTensor*>(orig[i]));
    this->tmp_indices.push_back(tmp);
  }

  auto indices = ExpandTensors<T, Context>(dev_ctx, this->tmp_indices);
  indices = ExpandOutplace<T, Context>(dev_ctx, indices);
  while (indices.size() < static_cast<size_t>(self.dims().size())) {
    indices.emplace_back();
  }

  std::vector<phi::DenseTensor*> indices_int64;
  for (auto& indice : indices) {
    if (indice && indice->dtype() == paddle::DataType::INT32) {
      *indice = phi::Cast<int, Context>(dev_ctx, *indice, phi::DataType::INT64);
    }
    indices_int64.push_back(indice);
  }

  phi::DenseTensor src = self;
  std::vector<phi::DenseTensor*> indices_list = indices_int64;

  uint32_t element_size_bytes = phi::SizeOf(src.dtype());
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  std::vector<int64_t> shape_vec = common::vectorize<int64_t>(src.dims());
  std::vector<int64_t> stride_vec = common::vectorize<int64_t>(src.strides());
  std::vector<int64_t> replacement_shape;
  std::vector<int64_t> idx_shape_vec = {};
  std::vector<int64_t> idx_stride_vec = {};
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim]) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = common::vectorize<int64_t>(indices_list[dim]->dims());

      indexed_sizes.push_back(shape_vec[dim]);
      indexed_strides.push_back(stride_vec[dim] * element_size_bytes);
    }
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = RestrideSrc(&src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index) {
      ReshapeIndexer(index, dims_before, dims_after);
      this->indices.push_back(index);
    }
  }
}

}  // namespace funcs
}  // namespace phi
